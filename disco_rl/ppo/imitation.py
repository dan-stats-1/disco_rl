"""Policy-only imitation of a frozen PPO teacher by the existing Disco agent."""

from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path

import dm_env
import haiku as hk
import jax
import jax.numpy as jnp
from jax.scipy.special import ndtri
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from ml_collections import config_dict
import numpy as np
import optax
import pandas as pd

from disco_rl import agent as agent_lib
from disco_rl import continuous_policies
from disco_rl import types
from disco_rl.environments import jittable_envs
from disco_rl.environments.rollouts import unroll_jittable_actor
from disco_rl.ppo.training import PPOResult
from disco_rl.ppo.training import PPOTrainer
from disco_rl.ppo.training import _finite_metrics


@dataclass(frozen=True)
class ImitationConfig:
  teacher_seed: int = 302
  seeds: tuple[int, ...] = (401, 402, 403)
  num_updates: int = 500
  num_envs: int = 8
  rollout_len: int = 32
  updates_per_rollout: int = 1
  eval_every: int = 50
  eval_num_envs: int = 16
  eval_steps: int = 256

  def __post_init__(self) -> None:
    sizes = (
        self.num_updates, self.num_envs, self.rollout_len,
        self.updates_per_rollout, self.eval_every,
        self.eval_num_envs, self.eval_steps,
    )
    if not self.seeds or any(
        not isinstance(size, int) or size < 1 for size in sizes
    ) or self.rollout_len < 2:
      raise ValueError('Use positive experiment sizes and rollout_len >= 2.')


class ImitationTrainer:
  """Applies only policy-matching gradients; no value or meta-updates run."""

  def __init__(
      self, teacher: PPOResult, agent_settings: config_dict.ConfigDict,
      config: ImitationConfig,
  ):
    if config.teacher_seed not in teacher.final_params:
      raise ValueError(f'No trained PPO teacher for seed {config.teacher_seed}.')
    self.teacher = teacher.trainer
    self.teacher_params = jax.tree.map(
        jax.lax.stop_gradient, teacher.final_params[config.teacher_seed]
    )
    self.config = config
    self.env = self.teacher.env_factory(config.num_envs)
    self.eval_env = self.teacher.env_factory(config.eval_num_envs)
    if not isinstance(self.env, jittable_envs.ContinuousCatchJittableEnvironment):
      raise ValueError('This imitation diagnostic requires continuous Catch.')
    spec = self.env.single_action_spec()
    if (
        float(spec.minimum) != self.teacher.action_low
        or float(spec.maximum) != self.teacher.action_high
    ):
      raise ValueError('Teacher and agent action bounds must match.')
    settings = config_dict.ConfigDict(agent_settings.to_dict())
    for cost in ('y_cost', 'z_cost', 'aux_policy_cost', 'value_cost'):
      settings.hyper_params[cost] = 0.0
    if not np.isfinite(settings.hyper_params.pi_cost) or settings.hyper_params.pi_cost <= 0:
      raise ValueError('pi_cost must be finite and positive.')
    self.agent = agent_lib.Agent(
        single_observation_spec=self.env.single_observation_spec(),
        single_action_spec=spec, agent_settings=settings, batch_axis_name=None,
    )
    self.num_samples = settings.update_rule.num_samples
    self.smoothing = settings.update_rule.wasserstein_smoothing
    self.collect = jax.jit(partial(
        unroll_jittable_actor, env=self.env,
        rollout_len=config.rollout_len, actor_step_fn=self.agent.actor_step,
    ))
    self.learn = jax.jit(self._learn)
    self.evaluate = jax.jit(self._evaluate, static_argnames=('mean_action',))

  def teacher_samples(self, obs: jax.Array) -> jax.Array:
    """A deterministic quantile bank of the frozen, bounded Gaussian policy."""
    mean, log_std, _ = self.teacher.network.apply(self.teacher_params, obs)
    quantiles = ndtri(
        (jnp.arange(self.num_samples, dtype=jnp.float32) + 0.5)
        / self.num_samples
    )
    samples = self.teacher.bounded_action(
        mean[..., None] + jnp.exp(log_std) * quantiles
    )
    return jax.lax.stop_gradient(samples)

  def policy_loss(
      self, params: hk.Params, actor_start: types.HaikuState,
      rollout: types.ActorRollout, targets: jax.Array,
  ):
    out, state = self.agent.unroll_net(params, actor_start, rollout)
    samples = out['logits'][:-1]
    mask = (rollout.discounts[:-1] > 0).astype(jnp.float32)
    denominator = mask.sum() + 1e-8
    per_step = continuous_policies.smooth_wasserstein_1(
        samples, jax.lax.stop_gradient(targets), self.smoothing
    )
    pi_loss = (per_step * mask).sum() / denominator
    total_loss = self.agent.settings.hyper_params.pi_cost * pi_loss
    metrics = dict(
        total_loss=total_loss, pi_loss=pi_loss,
        policy_spread=(samples.std(axis=-1) * mask).sum() / denominator,
        teacher_spread=(targets.std(axis=-1) * mask).sum() / denominator,
        valid_transitions=mask.sum(),
    )
    return total_loss, (state, metrics)

  def _learn(
      self, learner: agent_lib.LearnerState,
      actor_start: types.HaikuState, rollout: types.ActorRollout,
  ):
    targets = self.teacher_samples(rollout.observations['observation'][:-1])
    _, (_, before) = self.policy_loss(
        learner.params, actor_start, rollout, targets
    )

    def update(carry, _):
      params, opt_state = carry
      (_, (_, metrics)), grads = jax.value_and_grad(
          self.policy_loss, has_aux=True
      )(params, actor_start, rollout, targets)
      updates, opt_state = self.agent._optimizer.update(grads, opt_state, params)
      metrics = dict(
          metrics, gradient_norm=optax.global_norm(grads),
          update_norm=optax.global_norm(updates),
      )
      return (optax.apply_updates(params, updates), opt_state), metrics

    (params, opt_state), step_metrics = jax.lax.scan(
        update, (learner.params, learner.opt_state), None,
        length=self.config.updates_per_rollout,
    )
    _, (actor_state, after) = self.policy_loss(
        params, actor_start, rollout, targets
    )
    metrics = dict(
        after, pi_loss_before=before['pi_loss'],
        gradient_norm=step_metrics['gradient_norm'].mean(),
        update_norm=step_metrics['update_norm'].mean(),
    )
    return agent_lib.LearnerState(
        params=params, opt_state=opt_state, meta_state=learner.meta_state
    ), actor_state, metrics

  def _evaluate(
      self, params: hk.Params, key: jax.Array, mean_action: bool,
  ):
    reset_key, scan_key = jax.random.split(key)
    state_key = jax.random.fold_in(key, 1)
    state, timestep = self.eval_env.reset(reset_key)
    actor_state = self.agent.initial_actor_state(state_key)

    def step(carry, step_key):
      state, timestep, actor_state, running_return = carry
      actor_out, actor_state = self.agent.actor_step(
          params, step_key, timestep, actor_state
      )
      samples = actor_out.agent_outs['logits']
      targets = self.teacher_samples(timestep.observation)
      distance = continuous_policies.smooth_wasserstein_1(
          samples, targets, self.smoothing
      ).sum()
      actions = samples.mean(axis=-1) if mean_action else actor_out.actions
      state, timestep = self.eval_env.step(state, actions)
      running_return = running_return + timestep.reward
      done = timestep.step_type == dm_env.StepType.LAST
      stats = (
          jnp.where(done, running_return, 0.0).sum(), done.sum(),
          (done & (running_return > 0.0)).sum(), distance,
      )
      return (
          state, timestep, actor_state,
          jnp.where(done, 0.0, running_return),
      ), stats

    _, stats = jax.lax.scan(
        step, (
            state, timestep, actor_state, jnp.zeros(self.config.eval_num_envs)
        ), jax.random.split(scan_key, self.config.eval_steps),
    )
    return jax.tree.map(jnp.sum, stats)

  def evaluation_row(
      self, params: hk.Params, key: jax.Array, seed: int,
      update: int, mean_action: bool,
  ) -> dict:
    reward_sum, episodes, catches, distance = jax.device_get(
        self.evaluate(params, key, mean_action)
    )
    if int(episodes) == 0:
      raise RuntimeError('No evaluation episodes completed; increase eval_steps.')
    row = dict(
        seed=seed, update=update,
        env_steps=update * self.config.rollout_len * self.config.num_envs,
        mode='student_mean_action' if mean_action else 'student_stochastic',
        mean_return=float(reward_sum) / int(episodes),
        catch_rate=int(catches) / int(episodes), episodes=int(episodes),
        pi_loss=float(distance) / (self.config.eval_steps * self.config.eval_num_envs),
    )
    if not np.isfinite([row['mean_return'], row['catch_rate'], row['pi_loss']]).all():
      raise FloatingPointError('Non-finite imitation evaluation metrics.')
    return row


@dataclass
class ImitationResult:
  trainer: ImitationTrainer
  training: pd.DataFrame
  evaluation: pd.DataFrame
  teacher_evaluation: pd.DataFrame
  final_params: dict[int, hk.Params]


def _rollout_rewards(rollout, next_timestep, running_return):
  rewards = jnp.concatenate((rollout.rewards[1:], next_timestep.reward[None]))
  done = jnp.concatenate((
      rollout.discounts[1:] == 0,
      (next_timestep.step_type == dm_env.StepType.LAST)[None],
  ))

  def step(total, transition):
    reward, terminal = transition
    total = total + reward
    stats = (
        jnp.where(terminal, total, 0.0).sum(), terminal.sum(),
        (terminal & (total > 0)).sum(),
    )
    return jnp.where(terminal, 0.0, total), stats

  running_return, stats = jax.lax.scan(step, running_return, (rewards, done))
  return running_return, jax.tree.map(jnp.sum, stats)


def train_imitation(
    teacher: PPOResult, agent_settings: config_dict.ConfigDict,
    config: ImitationConfig = ImitationConfig(),
) -> ImitationResult:
  """Learn only pi on fresh student rollouts; PPO and meta-state remain fixed."""
  trainer = ImitationTrainer(teacher, agent_settings, config)
  teacher_rows, eval_rows, train_rows = [], [], []
  final_params = {}
  reward_stats = jax.jit(_rollout_rewards)
  teacher_eval = PPOTrainer(
      teacher.trainer.env_factory, replace(
          teacher.trainer.config, eval_num_envs=config.eval_num_envs,
          eval_steps=config.eval_steps,
      )
  )
  for seed in config.seeds:
    learner_key, actor_key, reset_key, key, eval_key = jax.random.split(
        jax.random.PRNGKey(seed), 5
    )
    learner = trainer.agent.initial_learner_state(learner_key)
    actor_state = trainer.agent.initial_actor_state(actor_key)
    env_state, timestep = trainer.env.reset(reset_key)
    running_return = jnp.zeros(config.num_envs)
    for mode in ('deterministic', 'stochastic'):
      row = teacher_eval.evaluation_row(
          trainer.teacher_params, eval_key, seed, 0, mode
      )
      row['mode'] = 'teacher_' + mode
      row['teacher_seed'] = config.teacher_seed
      teacher_rows.append(row)
    for mean_action in (False, True):
      eval_rows.append(trainer.evaluation_row(
          learner.params, eval_key, seed, 0, mean_action
      ))

    for update in range(1, config.num_updates + 1):
      key, collect_key = jax.random.split(key)
      actor_start = actor_state
      rollout, _, timestep, env_state = trainer.collect(
          learner.params, actor_start, timestep, env_state, collect_key
      )
      learner, actor_state, metrics = trainer.learn(learner, actor_start, rollout)
      scalars = _finite_metrics(metrics, f'imitation seed={seed}, update={update}')
      if scalars['valid_transitions'] == 0:
        raise RuntimeError('Rollout contains no valid policy-training positions.')
      running_return, stats = reward_stats(rollout, timestep, running_return)
      reward_sum, episodes, catches = jax.device_get(stats)
      if int(episodes) == 0:
        raise RuntimeError('No training episodes completed; increase rollout_len.')
      train_rows.append(dict(
          seed=seed, update=update,
          env_steps=update * config.rollout_len * config.num_envs,
          optimiser_steps=update * config.updates_per_rollout,
          mean_return=float(reward_sum) / int(episodes),
          catch_rate=int(catches) / int(episodes), episodes=int(episodes),
          **scalars,
      ))
      if update % config.eval_every == 0 or update == config.num_updates:
        for mean_action in (False, True):
          eval_rows.append(trainer.evaluation_row(
              learner.params, eval_key, seed, update, mean_action
          ))
        print(
            f"seed={seed} steps={train_rows[-1]['env_steps']:,} "
            f"held-out W={eval_rows[-2]['pi_loss']:.5f}, "
            f"catch={eval_rows[-2]['catch_rate']:.1%} (sampled), "
            f"{eval_rows[-1]['catch_rate']:.1%} (mean action)"
        )
    final_params[seed] = learner.params

  result = ImitationResult(
      trainer=trainer, training=pd.DataFrame(train_rows),
      evaluation=pd.DataFrame(eval_rows),
      teacher_evaluation=pd.DataFrame(teacher_rows), final_params=final_params,
  )
  print('\nFrozen PPO teacher:')
  print(result.teacher_evaluation.to_string(index=False))
  print('\nStudent before and after policy-only imitation:')
  print(result.evaluation[
      result.evaluation['update'].isin((0, config.num_updates))
  ].to_string(index=False))
  print('Only pi loss is used. PPO weights, meta-state, and y/z/p/q objectives are not updated.')
  return result


def save_imitation_results(result: ImitationResult, output_dir: Path) -> Figure:
  fig, axes = plt.subplots(1, 3, figsize=(17, 4))
  for seed, group in result.training.groupby('seed'):
    axes[0].plot(
        group.env_steps, group.pi_loss, label=f'seed {seed} (after update)'
    )
  axes[0].set(
      title='Policy-only loss on student rollouts',
      xlabel='Environment transitions', ylabel='Smooth Wasserstein',
      yscale='symlog',
  )
  for ax, metric in zip(axes[1:], ('mean_return', 'catch_rate')):
    for mode in ('student_stochastic', 'student_mean_action'):
      summary = result.evaluation[
          result.evaluation['mode'] == mode
      ].groupby('env_steps')[metric].agg(['mean', 'std'])
      std = summary['std'].fillna(0.0)
      ax.plot(summary.index, summary['mean'], label=mode)
      ax.fill_between(
          summary.index, summary['mean'] - std,
          summary['mean'] + std, alpha=0.15,
      )
    for mode, linestyle in (
        ('teacher_stochastic', '--'), ('teacher_deterministic', ':')
    ):
      ax.axhline(
          result.teacher_evaluation[
              result.teacher_evaluation['mode'] == mode
          ][metric].mean(), linestyle=linestyle, label=mode,
      )
    ax.set(
        title=f'Held-out {metric}', xlabel='Environment transitions',
        ylabel=metric,
    )
  for ax in axes:
    ax.legend(fontsize=8)
  fig.suptitle(
      'Disco agent imitating frozen PPO using only policy Wasserstein loss'
  )
  fig.tight_layout()
  output_dir.mkdir(parents=True, exist_ok=True)
  result.training.to_csv(output_dir / 'training.csv', index=False)
  result.evaluation.to_csv(output_dir / 'evaluation.csv', index=False)
  result.teacher_evaluation.to_csv(output_dir / 'teacher_evaluation.csv', index=False)
  fig.savefig(output_dir / 'learning_curves.png', dpi=150)
  print(f'Imitation histories and plot saved to {output_dir}')
  return fig
