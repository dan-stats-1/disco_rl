"""Clipped PPO with GAE and a tanh-bounded scalar Gaussian policy."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import dm_env
import haiku as hk
import jax
import jax.numpy as jnp
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import optax
import pandas as pd

from disco_rl.environments import base
from disco_rl.environments.wrappers.batched_jittable_env import EnvState

EnvFactory = Callable[[int], base.Environment]
Batch = dict[str, jax.Array]


@dataclass(frozen=True)
class PPOConfig:
  seeds: tuple[int, ...] = (301, 302, 303)
  num_updates: int = 100
  num_envs: int = 32
  horizon: int = 128
  epochs: int = 4
  num_minibatches: int = 8
  learning_rate: float = 3e-4
  gamma: float = 0.99
  gae_lambda: float = 0.95
  clip: float = 0.2
  value_cost: float = 0.5
  eval_every: int = 10
  eval_num_envs: int = 64
  eval_steps: int = 140
  target_catch_rate: float = 0.95
  checkpoint_steps: tuple[int, ...] = (24_576,)

  def __post_init__(self) -> None:
    sizes = (
        self.num_updates, self.num_envs, self.horizon, self.epochs,
        self.num_minibatches, self.eval_every, self.eval_num_envs, self.eval_steps,
    )
    if (
        not self.seeds
        or any(not isinstance(size, int) or size < 1 for size in sizes)
        or self.batch_size % self.num_minibatches
    ):
      raise ValueError(
          'Use positive sizes and a rollout batch divisible by minibatches.'
      )
    if any(
        not isinstance(steps, int) or steps < 0 or steps % self.batch_size
        for steps in self.checkpoint_steps
    ):
      raise ValueError('Checkpoint steps must be nonnegative completed updates.')

  @property
  def batch_size(self) -> int:
    return self.num_envs * self.horizon

  @property
  def total_steps(self) -> int:
    return self.num_updates * self.batch_size


def _network_fn(obs: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
  actor = hk.nets.MLP(
      (64, 64), activation=jnp.tanh, activate_final=True, name='actor'
  )(obs)
  critic = hk.nets.MLP(
      (64, 64), activation=jnp.tanh, activate_final=True, name='critic'
  )(obs)
  mean = hk.Linear(
      1, w_init=hk.initializers.RandomNormal(0.01), name='mean'
  )(actor)[..., 0]
  value = hk.Linear(1, name='value')(critic)[..., 0]
  log_std = hk.get_parameter(
      'log_std', shape=(), init=hk.initializers.Constant(-0.5)
  )
  return mean, jnp.clip(log_std, -5.0, 2.0), value


def _normal_log_prob(
    raw_action: jax.Array, mean: jax.Array, log_std: jax.Array
) -> jax.Array:
  return -0.5 * (
      ((raw_action - mean) * jnp.exp(-log_std)) ** 2
      + 2.0 * log_std + jnp.log(2.0 * jnp.pi)
  )


def _finite_metrics(metrics: Batch, context: str) -> dict[str, float]:
  scalars = {
      name: float(value) for name, value in jax.device_get(metrics).items()
  }
  if not np.isfinite(list(scalars.values())).all():
    raise FloatingPointError(f'Non-finite PPO metrics: {context}.')
  return scalars


class PPOTrainer:
  """Owns the environment, compiled operations, and settings of one PPO run."""

  def __init__(self, env_factory: EnvFactory, config: PPOConfig):
    self.env_factory = env_factory
    self.config = config
    self.env = env_factory(config.num_envs)
    self.eval_env = env_factory(config.eval_num_envs)
    spec = self.env.single_action_spec()
    if spec.shape or not np.issubdtype(spec.dtype, np.floating):
      raise ValueError('PPO requires a bounded scalar continuous action.')
    self.action_low = float(spec.minimum)
    self.action_high = float(spec.maximum)
    if (
        not np.isfinite([self.action_low, self.action_high]).all()
        or self.action_low >= self.action_high
    ):
      raise ValueError('PPO requires finite, ordered action bounds.')
    self.network = hk.without_apply_rng(hk.transform(_network_fn))
    self.optimizer = optax.chain(
        optax.clip_by_global_norm(0.5),
        optax.adam(config.learning_rate, eps=1e-5),
    )
    self.collect = jax.jit(self._collect)
    self.update = jax.jit(self._update)
    self.evaluate = jax.jit(self._evaluate, static_argnames=('mode',))

  def bounded_action(self, raw_action: jax.Array) -> jax.Array:
    midpoint = (self.action_high + self.action_low) / 2.0
    scale = (self.action_high - self.action_low) / 2.0
    return midpoint + scale * jnp.tanh(raw_action)

  def initial_state(
      self, seed: int
  ) -> tuple[hk.Params, EnvState, jax.Array, jax.Array, jax.Array]:
    init_key, reset_key, key, eval_key = jax.random.split(
        jax.random.PRNGKey(seed), 4
    )
    state, timestep = self.env.reset(reset_key)
    obs = timestep.observation
    params = self.network.init(init_key, obs)
    return params, state, obs, key, eval_key

  def _collect(
      self, params: hk.Params, env_state: EnvState, obs: jax.Array,
      running_return: jax.Array, key: jax.Array,
  ) -> tuple[EnvState, jax.Array, jax.Array, Batch, tuple[jax.Array, ...]]:
    config = self.config

    def step(carry, step_key):
      env_state, obs, running_return = carry
      mean, log_std, value = self.network.apply(params, obs)
      raw = mean + jnp.exp(log_std) * jax.random.normal(step_key, mean.shape)
      log_prob = _normal_log_prob(raw, mean, log_std)
      env_state, timestep = self.env.step(env_state, self.bounded_action(raw))
      done = timestep.step_type == dm_env.StepType.LAST
      running_return = running_return + timestep.reward
      stats = (
          jnp.where(done, running_return, 0.0).sum(), done.sum(),
          (done & (running_return > 0.0)).sum(),
      )
      running_return = jnp.where(done, 0.0, running_return)
      transition = dict(
          obs=obs, raw_action=raw, log_prob=log_prob, value=value,
          reward=timestep.reward, done=done,
      )
      return (
          env_state, timestep.observation, running_return
      ), (transition, stats)

    (env_state, obs, running_return), (rollout, stats) = jax.lax.scan(
        step, (env_state, obs, running_return),
        jax.random.split(key, config.horizon),
    )
    _, _, bootstrap = self.network.apply(params, obs)

    def gae_step(carry, transition):
      advantage, next_value = carry
      reward, done, value = transition
      discount = config.gamma * (1.0 - done)
      delta = reward + discount * next_value - value
      advantage = delta + discount * config.gae_lambda * advantage
      return (advantage, value), advantage

    # LAST observations are reset; terminal masking prevents cross-episode bootstrap.
    _, advantages = jax.lax.scan(
        gae_step, (jnp.zeros_like(bootstrap), bootstrap),
        (rollout['reward'], rollout['done'], rollout['value']), reverse=True,
    )
    batch = dict(
        obs=rollout['obs'], raw_action=rollout['raw_action'],
        old_log_prob=rollout['log_prob'], advantage=advantages,
        returns=advantages + rollout['value'],
    )
    batch = jax.tree.map(
        lambda x: x.reshape((config.batch_size,) + x.shape[2:]), batch
    )
    batch['advantage'] = (
        batch['advantage'] - batch['advantage'].mean()
    ) / (batch['advantage'].std() + 1e-8)
    return (
        env_state, obs, running_return, jax.lax.stop_gradient(batch),
        jax.tree.map(jnp.sum, stats),
    )

  def loss(self, params: hk.Params, batch: Batch) -> tuple[jax.Array, Batch]:
    mean, log_std, value = self.network.apply(params, batch['obs'])
    log_prob = _normal_log_prob(batch['raw_action'], mean, log_std)
    # The fixed tanh/scale Jacobian cancels at the same stored raw action.
    log_ratio = log_prob - batch['old_log_prob']
    ratio = jnp.exp(log_ratio)
    policy_loss = -jnp.minimum(
        ratio * batch['advantage'],
        jnp.clip(ratio, 1.0 - self.config.clip, 1.0 + self.config.clip)
        * batch['advantage'],
    ).mean()
    value_loss = 0.5 * jnp.square(value - batch['returns']).mean()
    return policy_loss + self.config.value_cost * value_loss, dict(
        policy_loss=policy_loss, value_loss=value_loss,
        approx_kl=((ratio - 1.0) - log_ratio).mean(),
        clip_fraction=(jnp.abs(ratio - 1.0) > self.config.clip).mean(),
        gaussian_std=jnp.exp(log_std),
    )

  def _update(
      self, params: hk.Params, opt_state: optax.OptState,
      batch: Batch, key: jax.Array,
  ) -> tuple[hk.Params, optax.OptState, Batch]:
    config = self.config

    def epoch(carry, epoch_key):
      params, opt_state = carry
      permutation = jax.random.permutation(epoch_key, config.batch_size)
      minibatches = jax.tree.map(
          lambda x: x[permutation].reshape((
              config.num_minibatches,
              config.batch_size // config.num_minibatches,
          ) + x.shape[1:]), batch,
      )

      def minibatch_step(carry, minibatch):
        params, opt_state = carry
        (loss, metrics), grads = jax.value_and_grad(
            self.loss, has_aux=True
        )(params, minibatch)
        updates, opt_state = self.optimizer.update(grads, opt_state, params)
        metrics = dict(
            metrics, loss=loss, gradient_norm=optax.global_norm(grads)
        )
        return (optax.apply_updates(params, updates), opt_state), metrics

      return jax.lax.scan(minibatch_step, (params, opt_state), minibatches)

    (params, opt_state), metrics = jax.lax.scan(
        epoch, (params, opt_state), jax.random.split(key, config.epochs)
    )
    return params, opt_state, jax.tree.map(jnp.mean, metrics)

  def _evaluate(
      self, params: hk.Params, key: jax.Array, mode: str
  ) -> tuple[jax.Array, ...]:
    if mode not in ('random', 'deterministic', 'stochastic'):
      raise ValueError(f'Unknown evaluation mode: {mode}')
    reset_key, scan_key = jax.random.split(key)
    env_state, timestep = self.eval_env.reset(reset_key)

    def step(carry, step_key):
      env_state, obs, running_return = carry
      mean, log_std, _ = self.network.apply(params, obs)
      if mode == 'random':
        actions = jax.random.uniform(
            step_key, mean.shape,
            minval=self.action_low, maxval=self.action_high,
        )
      elif mode == 'deterministic':
        actions = self.bounded_action(mean)
      else:
        actions = self.bounded_action(
            mean + jnp.exp(log_std) * jax.random.normal(step_key, mean.shape)
        )
      env_state, timestep = self.eval_env.step(env_state, actions)
      done = timestep.step_type == dm_env.StepType.LAST
      running_return = running_return + timestep.reward
      stats = (
          jnp.where(done, running_return, 0.0).sum(), done.sum(),
          (done & (running_return > 0.0)).sum(),
      )
      return (
          env_state, timestep.observation,
          jnp.where(done, 0.0, running_return),
      ), stats

    _, stats = jax.lax.scan(
        step, (
            env_state, timestep.observation,
            jnp.zeros(self.config.eval_num_envs),
        ), jax.random.split(scan_key, self.config.eval_steps),
    )
    return jax.tree.map(jnp.sum, stats)

  def evaluation_row(
      self, params: hk.Params, key: jax.Array, seed: int,
      update: int, mode: str,
  ) -> dict[str, int | float | str]:
    reward_sum, episodes, catches = jax.device_get(
        self.evaluate(params, key, mode)
    )
    if int(episodes) == 0:
      raise RuntimeError('No evaluation episodes completed; increase eval_steps.')
    mean_return = float(reward_sum) / int(episodes)
    if not np.isfinite(mean_return):
      raise FloatingPointError('Non-finite PPO evaluation return.')
    return dict(
        seed=seed, update=update, env_steps=update * self.config.batch_size,
        mode=mode, mean_return=mean_return,
        catch_rate=int(catches) / int(episodes), episodes=int(episodes),
    )

  def replay_checkpoints(self, seed: int, steps: int) -> dict[int, hk.Params]:
    if steps < 0 or steps % self.config.batch_size:
      raise ValueError('Replay steps must be a nonnegative completed update.')
    params, state, obs, key, _ = self.initial_state(seed)
    checkpoints = {0: params}
    opt_state = self.optimizer.init(params)
    running_return = jnp.zeros(self.config.num_envs)
    for update in range(steps // self.config.batch_size):
      key, collect_key, update_key = jax.random.split(key, 3)
      state, obs, running_return, batch, _ = self.collect(
          params, state, obs, running_return, collect_key
      )
      params, opt_state, metrics = self.update(
          params, opt_state, batch, update_key
      )
      _finite_metrics(metrics, f'replaying seed={seed}, update={update + 1}')
    checkpoints[steps] = params
    return checkpoints


@dataclass
class PPOResult:
  trainer: PPOTrainer
  training: pd.DataFrame
  evaluation: pd.DataFrame
  final_params: dict[int, hk.Params]
  checkpoints: dict[int, dict[int, hk.Params]]

  def checkpoint(self, seed: int, steps: int) -> hk.Params:
    if seed not in self.final_params:
      raise ValueError(f'No trained policy for seed {seed}.')
    if (
        steps < 0 or steps > self.trainer.config.total_steps
        or steps % self.trainer.config.batch_size
    ):
      raise ValueError('Checkpoint steps must be completed updates within this run.')
    saved = self.checkpoints[seed]
    if steps not in saved:
      print(
          f'Reconstructing the first {steps // self.trainer.config.batch_size} '
          f'updates of seed {seed}; the fully trained policy is unchanged.'
      )
      saved.update(self.trainer.replay_checkpoints(seed, steps))
    return saved[steps]


def train_ppo(
    env_factory: EnvFactory, config: PPOConfig = PPOConfig()
) -> PPOResult:
  """Train independent seeds without modifying any notebook or Disco state."""
  trainer = PPOTrainer(env_factory, config)
  eval_rows, train_rows = [], []
  final_params, checkpoints = {}, {}
  for seed in config.seeds:
    params, env_state, obs, key, eval_key = trainer.initial_state(seed)
    checkpoints[seed] = {0: params}
    opt_state = trainer.optimizer.init(params)
    running_return = jnp.zeros(config.num_envs)
    for mode in ('random', 'deterministic', 'stochastic'):
      eval_rows.append(trainer.evaluation_row(params, eval_key, seed, 0, mode))

    for update in range(1, config.num_updates + 1):
      key, collect_key, update_key = jax.random.split(key, 3)
      env_state, obs, running_return, batch, episode_stats = trainer.collect(
          params, env_state, obs, running_return, collect_key
      )
      params, opt_state, metrics = trainer.update(
          params, opt_state, batch, update_key
      )
      steps = update * config.batch_size
      if steps in config.checkpoint_steps:
        checkpoints[seed][steps] = params
      scalars = _finite_metrics(metrics, f'seed={seed}, update={update}')
      reward_sum, episodes, catches = jax.device_get(episode_stats)
      if int(episodes) == 0:
        raise RuntimeError('No training episodes completed; increase horizon.')
      train_rows.append(dict(
          seed=seed, update=update, env_steps=steps, episodes=int(episodes),
          mean_return=float(reward_sum) / int(episodes),
          catch_rate=int(catches) / int(episodes), **scalars,
      ))
      if update % config.eval_every == 0 or update == config.num_updates:
        for mode in ('deterministic', 'stochastic'):
          eval_rows.append(
              trainer.evaluation_row(params, eval_key, seed, update, mode)
          )
        print(
            f'seed={seed} steps={steps:,} eval catch: '
            f"deterministic={eval_rows[-2]['catch_rate']:.1%}, "
            f"stochastic={eval_rows[-1]['catch_rate']:.1%}"
        )
    final_params[seed] = params
    checkpoints[seed][config.total_steps] = params

  result = PPOResult(
      trainer=trainer, training=pd.DataFrame(train_rows),
      evaluation=pd.DataFrame(eval_rows), final_params=final_params,
      checkpoints=checkpoints,
  )
  print('\nInitial baselines and final held-out evaluations:')
  print(result.evaluation[
      result.evaluation['update'].isin((0, config.num_updates))
  ].to_string(index=False))
  final = result.evaluation[
      (result.evaluation['update'] == config.num_updates)
      & (result.evaluation['mode'] == 'deterministic')
  ]
  solved = (final.catch_rate >= config.target_catch_rate).all()
  print(
      f'All seeds reach {config.target_catch_rate:.0%} '
      f'deterministic catch rate: {solved}'
  )
  if not solved:
    print(
        'Threshold not reached in this run; PPO failure alone does not prove '
        'the environment unsolvable.'
    )
  return result


def save_training_results(result: PPOResult, output_dir: Path) -> Figure:
  """Save reward histories and return the learning-curve figure for display."""
  fig, axes = plt.subplots(1, 3, figsize=(17, 4))
  for seed, group in result.training.groupby('seed'):
    axes[0].plot(
        group.env_steps, group.mean_return.rolling(5, min_periods=1).mean(),
        label=f'seed {seed}',
    )
  axes[0].set(
      title='Training episode return (5-update smoothing)',
      xlabel='Environment transitions', ylabel='Mean return',
  )
  for ax, metric in zip(axes[1:], ('mean_return', 'catch_rate')):
    for mode in ('deterministic', 'stochastic'):
      summary = result.evaluation[
          result.evaluation['mode'] == mode
      ].groupby('env_steps')[metric].agg(['mean', 'std'])
      std = summary['std'].fillna(0.0)
      ax.plot(summary.index, summary['mean'], label=mode)
      ax.fill_between(
          summary.index, summary['mean'] - std,
          summary['mean'] + std, alpha=0.15,
      )
    ax.axhline(
        result.evaluation[
            result.evaluation['mode'] == 'random'
        ][metric].mean(),
        color='black', linestyle='--', label='Uniform random actions',
    )
    ax.set(
        title=f'Held-out {metric}',
        xlabel='Environment transitions', ylabel=metric,
    )
  for ax in axes:
    ax.legend()
  fig.suptitle(
      'PPO on unchanged continuous Catch; '
      'shading = across-seed standard deviation'
  )
  fig.tight_layout()
  output_dir.mkdir(parents=True, exist_ok=True)
  result.training.to_csv(output_dir / 'training.csv', index=False)
  result.evaluation.to_csv(output_dir / 'evaluation.csv', index=False)
  fig.savefig(output_dir / 'learning_curves.png', dpi=150)
  print(f'Reward histories and plot saved to {output_dir}')
  return fig
