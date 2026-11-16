"""Discover Disco targets using a frozen PPO outer objective."""

from dataclasses import dataclass
import math
from pathlib import Path

import haiku as hk
import jax
import jax.numpy as jnp
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from ml_collections import config_dict
import optax
import pandas as pd

from disco_rl import agent as agent_lib
from disco_rl import types
from disco_rl import utils
from disco_rl.ppo.imitation import ImitationConfig
from disco_rl.ppo.imitation import ImitationTrainer
from disco_rl.ppo.training import PPOResult
from disco_rl.ppo.training import _finite_metrics


@dataclass(frozen=True)
class MetaTrainingConfig:
  teacher_seed: int = 302
  meta_seed: int = 501
  seeds: tuple[int, ...] = (511, 512)
  num_updates: int = 500
  num_inner_steps: int = 4
  num_envs: int = 2
  rollout_len: int = 8
  meta_learning_rate: float = 1e-3
  eval_every: int = 50
  eval_num_envs: int = 16
  eval_steps: int = 140
  fresh_seeds: tuple[int, ...] = (521, 522, 523)
  fresh_updates: int = 100
  use_auxiliary_losses: bool = False

  def __post_init__(self) -> None:
    sizes = (
        self.num_updates, self.num_inner_steps, self.num_envs,
        self.rollout_len, self.eval_every, self.eval_num_envs,
        self.eval_steps, self.fresh_updates,
    )
    if (
        not self.seeds or not self.fresh_seeds
        or len(set(self.seeds)) != len(self.seeds)
        or len(set(self.fresh_seeds)) != len(self.fresh_seeds)
        or set(self.seeds) & set(self.fresh_seeds)
        or any(not isinstance(n, int) or n < 1 for n in sizes)
        or self.rollout_len < 2
    ):
      raise ValueError('Use positive sizes and distinct training/fresh seeds.')
    if not math.isfinite(self.meta_learning_rate) or self.meta_learning_rate <= 0:
      raise ValueError('meta_learning_rate must be finite and positive.')

  @property
  def transitions_per_update(self) -> int:
    # Validation collection is twice as long as each inner-training rollout.
    return len(self.seeds) * (self.num_inner_steps + 2) * self.rollout_len * self.num_envs


class MetaTrainingTrainer:
  """PPO judges updated students; only Disco supplies their inner targets."""

  def __init__(
      self, teacher: PPOResult, agent_settings: config_dict.ConfigDict,
      config: MetaTrainingConfig,
  ):
    self.config = config
    self.student = ImitationTrainer(teacher, agent_settings, ImitationConfig(
        teacher_seed=config.teacher_seed, seeds=config.seeds,
        num_updates=config.num_updates, num_envs=config.num_envs,
        rollout_len=config.rollout_len, eval_every=config.eval_every,
        eval_num_envs=config.eval_num_envs, eval_steps=config.eval_steps,
    ))
    self.agent = self.student.agent
    if config.use_auxiliary_losses:
      for cost in ('y_cost', 'z_cost', 'aux_policy_cost', 'value_cost'):
        weight = agent_settings.hyper_params[cost]
        if not math.isfinite(weight) or weight <= 0:
          raise ValueError(f'{cost} must be finite and positive for full inner learning.')
        # Restore only this trainer's copy, before any learner is traced.
        self.agent.settings.hyper_params[cost] = weight
    self.collect = self.student.collect
    self.collect_validation = jax.jit(
        lambda params, state, ts, env_state, key:
        self._collect_validation(params, state, ts, env_state, key)
    )
    self.meta_optimizer = optax.chain(
        optax.clip_by_global_norm(1.0), optax.adam(config.meta_learning_rate)
    )
    self.gradient = jax.jit(jax.value_and_grad(self.outer_loss, has_aux=True))
    self.apply_rule = jax.jit(self._apply_rule)

  def _collect_validation(self, params, state, ts, env_state, key):
    from disco_rl.environments.rollouts import unroll_jittable_actor

    return unroll_jittable_actor(
        params, state, ts, env_state, key, self.student.env,
        2 * self.config.rollout_len, self.agent.actor_step,
    )

  def _apply_rule(self, meta_params, learner, actor_start, rollout, key):
    return self.agent.learner_step(
        key, rollout, learner, actor_start, meta_params, is_meta_training=False
    )

  def outer_loss(
      self, meta_params: types.MetaParams, learner: agent_lib.LearnerState,
      train_rollouts: types.ActorRollout, validation: types.ActorRollout,
      key: jax.Array,
  ):
    """Differentiate validation policy distance through K agent updates."""
    def inner_step(carry, inputs):
      current, actor_state = carry
      rollout, step_key = inputs
      updated, actor_state, metrics = self.agent.learner_step(
          step_key, rollout, current, actor_state, meta_params,
          is_meta_training=True,
      )
      return (updated, actor_state), {
          name: metrics[name] for name in (
              'total_loss', 'global_gradient_norm', 'global_update_norm',
              'pi_loss', 'y_loss', 'z_loss', 'aux_kl_loss', 'q_loss',
          )
      }

    first_rollout = jax.tree.map(lambda x: x[0], train_rollouts)
    (updated, _), inner_metrics = jax.lax.scan(
        inner_step, (learner, first_rollout.first_state(time_axis=0)),
        (train_rollouts, jax.random.split(key, self.config.num_inner_steps)),
    )
    targets = self.student.teacher_samples(
        validation.observations['observation'][:-1]
    )
    loss, (_, metrics) = self.student.policy_loss(
        updated.params, validation.first_state(time_axis=0),
        validation, targets,
    )
    # pi_cost scales the inner objective only, not the outer comparison.
    loss = metrics['pi_loss']
    logs = dict(
        meta_loss=loss, valid_transitions=metrics['valid_transitions'],
        inner_loss=inner_metrics['total_loss'].mean(),
        agent_gradient_norm=inner_metrics['global_gradient_norm'].mean(),
        agent_update_norm=inner_metrics['global_update_norm'].mean(),
    )
    for name in ('pi_loss', 'y_loss', 'z_loss', 'aux_kl_loss', 'q_loss'):
      logs['inner_' + name] = inner_metrics[name].mean()
    return loss, (updated, logs)

  def evaluation_rows(self, params, seed, update, env_steps, stage):
    rows = []
    for mean_action in (False, True):
      row = self.student.evaluation_row(
          params, jax.random.PRNGKey(seed + 10000), seed, update, mean_action
      )
      row.update(env_steps=env_steps, stage=stage)
      rows.append(row)
    return rows


@dataclass
class MetaTrainingResult:
  trainer: MetaTrainingTrainer
  training: pd.DataFrame
  evaluation: pd.DataFrame
  fresh_evaluation: pd.DataFrame
  initial_meta_params: hk.Params
  final_meta_params: hk.Params
  final_params: dict[int, hk.Params]


def _initial_student(trainer, seed):
  learner_key, actor_key, reset_key, key = jax.random.split(
      jax.random.PRNGKey(seed), 4
  )
  learner = trainer.agent.initial_learner_state(learner_key)
  actor = trainer.agent.initial_actor_state(actor_key)
  env_state, ts = trainer.student.env.reset(reset_key)
  return learner, actor, ts, env_state, key


def train_meta(
    teacher: PPOResult, agent_settings: config_dict.ConfigDict,
    config: MetaTrainingConfig = MetaTrainingConfig(),
) -> MetaTrainingResult:
  """Truncated meta-gradients; experience collection is not differentiated."""
  trainer = MetaTrainingTrainer(teacher, agent_settings, config)
  meta_params, _ = trainer.agent.update_rule.init_params(
      jax.random.PRNGKey(config.meta_seed)
  )
  initial_meta_params = meta_params
  opt_state = trainer.meta_optimizer.init(meta_params)
  states = [_initial_student(trainer, seed) for seed in config.seeds]
  train_rows, eval_rows = [], []
  for seed, state in zip(config.seeds, states):
    eval_rows.extend(trainer.evaluation_rows(state[0].params, seed, 0, 0, 'meta_train'))

  for update in range(1, config.num_updates + 1):
    gradients, logs = [], []
    for index, (learner, actor, ts, env_state, key) in enumerate(states):
      rollouts = []
      for _ in range(config.num_inner_steps):
        key, collect_key = jax.random.split(key)
        rollout, actor, ts, env_state = trainer.collect(
            learner.params, actor, ts, env_state, collect_key
        )
        rollouts.append(rollout)
      key, valid_key, gradient_key = jax.random.split(key, 3)
      validation, actor, ts, env_state = trainer.collect_validation(
          learner.params, actor, ts, env_state, valid_key
      )
      (_, (learner, log)), grad = trainer.gradient(
          meta_params, learner, utils.tree_stack(rollouts), validation,
          gradient_key,
      )
      gradients.append(grad)
      logs.append(log)
      # Truncate history between outer iterations, including optimiser/RNN state.
      states[index] = jax.tree.map(
          jax.lax.stop_gradient, (learner, actor, ts, env_state, key)
      )
    grad = jax.tree.map(lambda *xs: jnp.stack(xs).mean(0), *gradients)
    updates, opt_state = trainer.meta_optimizer.update(grad, opt_state, meta_params)
    metrics = jax.tree.map(lambda *xs: jnp.stack(xs).mean(0), *logs)
    metrics.update(
        meta_gradient_norm=optax.global_norm(grad),
        meta_update_norm=optax.global_norm(updates),
    )
    scalars = _finite_metrics(metrics, f'PPO outer meta-update {update}')
    if scalars['valid_transitions'] == 0:
      raise RuntimeError('No valid outer-loss positions; increase rollout_len.')
    meta_params = optax.apply_updates(meta_params, updates)
    env_steps = update * config.transitions_per_update
    train_rows.append(dict(update=update, env_steps=env_steps, **scalars))
    if update % config.eval_every == 0 or update == config.num_updates:
      for seed, state in zip(config.seeds, states):
        eval_rows.extend(trainer.evaluation_rows(
            state[0].params, seed, update, env_steps, 'meta_train'
        ))
      print(
          f"meta-update={update} transitions={env_steps:,} "
          f"outer W={scalars['meta_loss']:.6f} "
          f"meta-gradient={scalars['meta_gradient_norm']:.6g}"
      )

  fresh_rows = []
  # PPO is used only for evaluation here, never for these students' updates.
  for seed in config.fresh_seeds:
    learner, actor, ts, env_state, key = _initial_student(trainer, seed)
    fresh_rows.extend(trainer.evaluation_rows(
        learner.params, seed, 0, 0, 'fresh_rule'
    ))
    for step in range(1, config.fresh_updates + 1):
      key, collect_key, learn_key = jax.random.split(key, 3)
      actor_start = actor
      rollout, actor, ts, env_state = trainer.collect(
          learner.params, actor, ts, env_state, collect_key
      )
      learner, _, metrics = trainer.apply_rule(
          meta_params, learner, actor_start, rollout, learn_key
      )
      _finite_metrics({
          name: metrics[name] for name in (
              'total_loss', 'global_gradient_norm', 'global_update_norm'
          )
      }, f'fresh rule seed={seed}, update={step}')
      if step % config.eval_every == 0 or step == config.fresh_updates:
        fresh_rows.extend(trainer.evaluation_rows(
            learner.params, seed, step,
            step * config.num_envs * config.rollout_len, 'fresh_rule',
        ))
  return MetaTrainingResult(
      trainer, pd.DataFrame(train_rows), pd.DataFrame(eval_rows),
      pd.DataFrame(fresh_rows), initial_meta_params, meta_params,
      {seed: state[0].params for seed, state in zip(config.seeds, states)},
  )


def save_meta_results(result: MetaTrainingResult, output_dir: Path) -> Figure:
  fig, axes = plt.subplots(1, 3, figsize=(17, 4))
  axes[0].plot(result.training.env_steps, result.training.meta_loss)
  axes[0].set(
      title='Outer PPO policy distance', xlabel='Meta-training transitions',
      ylabel='Smooth Wasserstein',
  )
  for ax, frame, title in (
      (axes[1], result.evaluation, 'Meta-training students'),
      (axes[2], result.fresh_evaluation, 'Fresh students: frozen learned rule'),
  ):
    for mode, group in frame.groupby('mode'):
      summary = group.groupby('env_steps').catch_rate.agg(['mean', 'std'])
      spread = summary['std'].fillna(0)
      ax.plot(summary.index, summary['mean'], label=mode)
      ax.fill_between(
          summary.index, summary['mean'] - spread,
          summary['mean'] + spread, alpha=0.15,
      )
    ax.set(title=title, xlabel='Environment transitions', ylabel='Catch rate', ylim=(0, 1))
    ax.legend(fontsize=8)
  fig.tight_layout()
  output_dir.mkdir(parents=True, exist_ok=True)
  result.training.to_csv(output_dir / 'training.csv', index=False)
  result.evaluation.to_csv(output_dir / 'evaluation.csv', index=False)
  result.fresh_evaluation.to_csv(output_dir / 'fresh_evaluation.csv', index=False)
  fig.savefig(output_dir / 'learning_curves.png', dpi=150)
  print(f'PPO-guided meta-training results saved to {output_dir}')
  return fig
