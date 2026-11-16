"""Measure real PPO outer-loss changes on one fixed inner-learning problem."""

from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Callable

import haiku as hk
import jax
import jax.numpy as jnp
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from ml_collections import config_dict
import numpy as np
import optax
import pandas as pd

from disco_rl import types
from disco_rl import utils
from disco_rl.ppo.meta_training import MetaTrainingConfig
from disco_rl.ppo.meta_training import MetaTrainingTrainer
from disco_rl.ppo.meta_training import _initial_student
from disco_rl.ppo.training import PPOResult


@dataclass
class MetaUpdateProbeResult:
  trials: pd.DataFrame
  slopes: pd.DataFrame
  figure: Figure
  baseline_loss: float
  gradient_norm: float
  replay_error: float
  fresh_adam_update_norm: float
  historical_adam_update_norm: float | None = None
  reference_loss_error: float | None = None
  direction_status: pd.DataFrame = field(default_factory=pd.DataFrame)
  gradient_nonzero_count: int = 0
  gradient_max_abs: float = 0.0


def _norm(tree) -> float:
  # Accumulate on the host in float64 so small float32 squares do not underflow.
  return float(np.sqrt(sum(
      np.square(np.asarray(x, dtype=np.float64)).sum()
      for x in jax.tree.leaves(tree)
  )))


def _dot(left, right) -> float:
  return float(sum(
      jnp.vdot(a, b) for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right))
  ))


def _finite_tree(tree, name):
  if not all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(tree)):
    raise FloatingPointError(f'Non-finite {name} in fixed-snapshot probe.')


def _measure_updates(
    loss_fn: Callable[[hk.Params], jax.Array], params: hk.Params,
    grads: hk.Params, optimizer: optax.GradientTransformation,
    step_norms: tuple[float, ...], loss_tolerance: float,
    *, meta_opt_state: optax.OptState | None = None,
) -> MetaUpdateProbeResult:
  if (
      not step_norms or any(not np.isfinite(s) or s <= 0 for s in step_norms)
      or len(set(step_norms)) != len(step_norms)
      or not np.isfinite(loss_tolerance) or loss_tolerance <= 0
  ):
    raise ValueError('Use distinct positive step norms and a positive loss tolerance.')
  _finite_tree((params, grads), 'parameters/gradients')
  gradient_norm = _norm(grads)
  gradient_nonzero_count = sum(
      int(np.count_nonzero(np.asarray(x))) for x in jax.tree.leaves(grads)
  )
  gradient_max_abs = max(
      (float(np.max(np.abs(np.asarray(x)), initial=0)) for x in jax.tree.leaves(grads)),
      default=0.0,
  )
  adam_update, _ = optimizer.update(grads, optimizer.init(params), params)
  _finite_tree(adam_update, 'fresh Adam update')
  adam_norm = _norm(adam_update)
  directions = {}
  statuses = []

  def add_direction(name, vector, norm):
    available = norm > 0
    statuses.append(dict(
        direction=name, unnormalised_norm=norm, tested=available,
        status='available' if available else 'unavailable_zero_norm',
    ))
    if available:
      directions[name] = jax.tree.map(lambda x: x / norm, vector)

  add_direction('negative_gradient', jax.tree.map(lambda g: -g, grads), gradient_norm)
  add_direction('fresh_adam', adam_update, adam_norm)
  historical_norm = None
  if meta_opt_state is not None:
    _finite_tree(meta_opt_state, 'historical Adam state')
    historical_update, _ = optimizer.update(grads, meta_opt_state, params)
    _finite_tree(historical_update, 'historical Adam proposal')
    historical_norm = _norm(historical_update)
    add_direction('historical_adam', historical_update, historical_norm)
  if gradient_norm == 0:
    print(
        'WARNING: Meta-gradient is exactly zero in the supplied arrays. '
        'Gradient-based directions are unavailable; this is not convergence. '
        'Testing two seeded random controls for measured loss sensitivity.'
    )
    leaves, structure = jax.tree.flatten(params)
    keys = jax.random.split(jax.random.PRNGKey(914), 2 * len(leaves))
    for i in range(2):
      control = jax.tree.unflatten(structure, [
          jax.random.normal(keys[i * len(leaves) + j], p.shape, p.dtype)
          for j, p in enumerate(leaves)
      ])
      add_direction(f'random_control_{i}', control, _norm(control))
  baseline = float(loss_fn(params))
  replay_error = abs(float(loss_fn(params)) - baseline)
  if not np.isfinite(baseline) or not np.isfinite(replay_error):
    raise FloatingPointError('Non-finite baseline outer loss.')
  if replay_error > loss_tolerance:
    raise RuntimeError('Baseline replay is not repeatable at the requested tolerance.')

  rows = [dict(
      direction='baseline', sign=0, requested_step_norm=0.0, actual_step_norm=0.0,
      outer_loss=baseline, actual_loss_change=0.0, predicted_loss_change=0.0,
      resolved_decrease=False, resolved_increase=False,
  )]
  slope_rows = []
  for name, direction in directions.items():
    predicted_slope = _dot(grads, direction)
    for size in sorted(step_norms):
      losses = {}
      for sign in (-1, 1):
        candidate = jax.tree.map(
            lambda p, d: p + sign * size * d, params, direction
        )
        perturbation = jax.tree.map(lambda p, q: p - q, candidate, params)
        value = float(loss_fn(candidate))
        if not np.isfinite(value):
          raise FloatingPointError(f'Non-finite outer loss for {name}, step={sign * size}.')
        change = value - baseline
        losses[sign] = value
        rows.append(dict(
            direction=name, sign=sign, requested_step_norm=size,
            actual_step_norm=_norm(perturbation), outer_loss=value,
            actual_loss_change=change,
            predicted_loss_change=_dot(grads, perturbation),
            resolved_decrease=change < -loss_tolerance,
            resolved_increase=change > loss_tolerance,
        ))
      slope_rows.append(dict(
          direction=name, step_norm=size, predicted_slope=predicted_slope,
          measured_central_slope=(losses[1] - losses[-1]) / (2 * size),
          slope_resolved=abs(losses[1] - losses[-1]) > 2 * loss_tolerance,
      ))
  trials, slopes = pd.DataFrame(rows), pd.DataFrame(slope_rows)
  fig, axes = plt.subplots(1, 2, figsize=(13, 4))
  for name in directions:
    for sign, label in ((1, 'update'), (-1, 'opposite control')):
      group = trials[(trials.direction == name) & (trials.sign == sign)]
      line, = axes[0].plot(
          group.requested_step_norm, group.actual_loss_change,
          marker='o', label=f'{name}: {label}',
      )
      axes[0].plot(
          group.requested_step_norm, group.predicted_loss_change,
          linestyle=':', color=line.get_color(),
      )
    group = slopes[slopes.direction == name]
    axes[1].plot(group.step_norm, group.measured_central_slope, marker='o', label=name)
    axes[1].axhline(group.predicted_slope.iloc[0], linestyle=':')
  axes[0].axhline(0, color='black', linewidth=0.8)
  axes[0].axhspan(-loss_tolerance, loss_tolerance, color='grey', alpha=0.15)
  axes[0].set(
      title='Actual change (solid) vs gradient prediction (dotted)',
      xlabel='Requested absolute parameter-step norm', ylabel='Outer loss change',
      xscale='log',
  )
  axes[1].set(
      title='Finite-difference slopes vs supplied meta-gradient',
      xlabel='Step norm', ylabel='Directional derivative', xscale='log',
  )
  for ax in axes:
    ax.legend(fontsize=7)
  fig.tight_layout()
  return MetaUpdateProbeResult(
      trials, slopes, fig, baseline, gradient_norm, replay_error, adam_norm,
      historical_norm, direction_status=pd.DataFrame(statuses),
      gradient_nonzero_count=gradient_nonzero_count,
      gradient_max_abs=gradient_max_abs,
  )


def _fixed_snapshot(
    teacher: PPOResult, agent_settings: config_dict.ConfigDict,
    config: MetaTrainingConfig, meta_params: types.MetaParams | None,
):
  if len(config.seeds) != 1:
    raise ValueError('The fixed-snapshot diagnostic uses exactly one student seed.')
  trainer = MetaTrainingTrainer(teacher, agent_settings, config)
  template, _ = trainer.agent.update_rule.init_params(
      jax.random.PRNGKey(config.meta_seed)
  )
  if meta_params is None:
    meta_params = template
  elif (
      jax.tree.structure(meta_params) != jax.tree.structure(template)
      or any(a.shape != b.shape for a, b in zip(
          jax.tree.leaves(meta_params), jax.tree.leaves(template)
      ))
  ):
    raise ValueError('Supplied meta-parameters do not match this agent/rule architecture.')
  params = jax.tree.map(jax.lax.stop_gradient, meta_params)
  return trainer, params, *_collect_fixed_snapshot(trainer, config.seeds[0])


def _collect_fixed_snapshot(trainer: MetaTrainingTrainer, seed: int):
  """Collect one reset learner/data problem, independently of rule parameters."""
  learner, actor, ts, env_state, key = _initial_student(trainer, seed)
  rollouts = []
  for _ in range(trainer.config.num_inner_steps):
    key, collect_key = jax.random.split(key)
    rollout, actor, ts, env_state = trainer.collect(
        learner.params, actor, ts, env_state, collect_key
    )
    rollouts.append(rollout)
  key, valid_key, learn_key = jax.random.split(key, 3)
  validation, _, _, _ = trainer.collect_validation(
      learner.params, actor, ts, env_state, valid_key
  )
  learner, train_rollouts, validation, learn_key = jax.tree.map(
      jax.lax.stop_gradient,
      (learner, utils.tree_stack(rollouts), validation, learn_key),
  )
  if not bool((validation.discounts[:-1] > 0).any()):
    raise RuntimeError('The validation snapshot contains no valid policy-loss positions.')
  return learner, train_rollouts, validation, learn_key


def probe_meta_updates(
    teacher: PPOResult, agent_settings: config_dict.ConfigDict,
    config: MetaTrainingConfig, *,
    meta_params: types.MetaParams | None = None,
    step_norms: tuple[float, ...] = (1e-5, 1e-4, 1e-3, 1e-2),
    loss_tolerance: float = 1e-7,
    meta_opt_state: optax.OptState | None = None,
    expected_loss: float | None = None,
    reference_tolerance: float = 1e-6,
) -> MetaUpdateProbeResult:
  """Collect once; replay identical theta/Adam/meta/RNN state for every eta."""
  if (
      not np.isfinite(reference_tolerance) or reference_tolerance <= 0
      or (expected_loss is not None and not np.isfinite(expected_loss))
  ):
    raise ValueError('Use a finite reference loss and positive reference tolerance.')
  trainer, params, learner, train_rollouts, validation, learn_key = _fixed_snapshot(
      teacher, agent_settings, config, meta_params
  )

  def outer_loss(eta):
    return trainer.outer_loss(
        eta, learner, train_rollouts, validation, learn_key
    )[0]

  # Only scalar losses escape replays; do not retain all post-update states.
  evaluate = jax.jit(outer_loss)
  baseline, grads = jax.jit(jax.value_and_grad(outer_loss))(params)
  reference_error = None
  if expected_loss is not None:
    reference_error = abs(float(evaluate(params)) - expected_loss)
    if not np.isfinite(reference_error) or reference_error > reference_tolerance:
      raise RuntimeError(
          f'Checkpoint replay loss does not match: error={reference_error:.3g}, '
          f'tolerance={reference_tolerance:.3g}. Restore the original teacher, '
          'agent settings, seeds and fixed-problem configuration.'
      )
  result = _measure_updates(
      evaluate, params, grads, trainer.meta_optimizer, step_norms, loss_tolerance,
      meta_opt_state=meta_opt_state,
  )
  result.reference_loss_error = reference_error
  if abs(float(baseline) - result.baseline_loss) > loss_tolerance:
    raise RuntimeError('Gradient and scalar-loss evaluations disagree on the baseline.')
  print(
      f'Fixed snapshot: {config.num_inner_steps} inner updates, '
      f'{config.rollout_len} rollout steps, {config.num_envs} environments, '
      f'{trainer.agent.settings.update_rule.num_samples} policy samples.'
  )
  print(
      f'Baseline outer W={result.baseline_loss:.8f}; '
      f'meta-gradient norm={result.gradient_norm:.6g}; '
      f'baseline replay error={result.replay_error:.3g}.'
  )
  print(
      f'Gradient entries: {result.gradient_nonzero_count} nonzero; '
      f'maximum absolute entry={result.gradient_max_abs:.6g}.'
  )
  print(result.direction_status.to_string(index=False))
  print('Step sizes are absolute L2 norms, not learning rates. Positive sign follows the named direction.')
  print('Adam starts from fresh outer optimiser state; it is not the historical training update.')
  if meta_opt_state is not None:
    print(
        f'Historical Adam is also tested without committing: '
        f'proposal norm={result.historical_adam_update_norm:.6g}.'
    )
  if reference_error is not None:
    print(f'Checkpoint reference loss error: {reference_error:.3g}.')
  print('No measurable decrease does not prove impossibility; inspect step resolution and surrogate-gradient slopes.')
  return result


def save_meta_update_probe(result: MetaUpdateProbeResult, output_dir: Path) -> None:
  output_dir.mkdir(parents=True, exist_ok=True)
  result.trials.to_csv(output_dir / 'trials.csv', index=False)
  result.slopes.to_csv(output_dir / 'directional_slopes.csv', index=False)
  result.direction_status.to_csv(output_dir / 'direction_status.csv', index=False)
  pd.DataFrame([dict(
      baseline_loss=result.baseline_loss, gradient_norm=result.gradient_norm,
      replay_error=result.replay_error,
      fresh_adam_update_norm=result.fresh_adam_update_norm,
      historical_adam_update_norm=result.historical_adam_update_norm,
      reference_loss_error=result.reference_loss_error,
      gradient_nonzero_count=result.gradient_nonzero_count,
      gradient_max_abs=result.gradient_max_abs,
  )]).to_csv(output_dir / 'summary.csv', index=False)
  result.figure.savefig(output_dir / 'loss_change.png', dpi=150)
  print(f'Fixed-snapshot meta-update measurements saved to {output_dir}')
