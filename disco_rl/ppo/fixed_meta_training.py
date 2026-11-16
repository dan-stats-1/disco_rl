"""Repeated meta-optimisation with a reset learner and fixed observation sets."""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import haiku as hk
import jax
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from ml_collections import config_dict
import numpy as np
import optax
import pandas as pd

from disco_rl import types
from disco_rl.ppo.meta_training import MetaTrainingConfig
from disco_rl.ppo.meta_update_validation import _dot
from disco_rl.ppo.meta_update_validation import _finite_tree
from disco_rl.ppo.meta_update_validation import _fixed_snapshot
from disco_rl.ppo.meta_update_validation import _norm
from disco_rl.ppo.meta_update_validation import MetaUpdateProbeResult
from disco_rl.ppo.meta_update_validation import probe_meta_updates
from disco_rl.ppo.training import PPOResult


@dataclass
class FixedMetaTrainingResult:
  training: pd.DataFrame
  step_trials: pd.DataFrame
  initial_meta_params: hk.Params
  final_meta_params: hk.Params
  final_meta_opt_state: optax.OptState
  replay_error: float
  applied_multiplier: float
  figure: Figure


def _optimise_fixed_loss(
    params: hk.Params,
    losses: Callable[[hk.Params], tuple[jax.Array, jax.Array]],
    optimizer: optax.GradientTransformation, num_updates: int,
    step_multipliers: tuple[float, ...], applied_multiplier: float,
    loss_tolerance: float, log_every: int,
    *, expected_initial_fit_loss: float | None = None,
    reference_tolerance: float = 1e-6,
    gradient_fn: Callable[
        [hk.Params], tuple[tuple[jax.Array, jax.Array], hk.Params]
    ] | None = None,
) -> FixedMetaTrainingResult:
  if (
      not isinstance(num_updates, int) or num_updates < 1
      or not isinstance(log_every, int) or log_every < 1
      or not step_multipliers
      or any(not np.isfinite(x) or x < 0 for x in step_multipliers)
      or not np.isfinite(applied_multiplier) or applied_multiplier <= 0
      or not np.isfinite(loss_tolerance) or loss_tolerance <= 0
      or not np.isfinite(reference_tolerance) or reference_tolerance <= 0
      or (expected_initial_fit_loss is not None and not np.isfinite(expected_initial_fit_loss))
  ):
    raise ValueError('Use positive update/log counts, finite nonnegative trial factors, positive applied factor/tolerances, and a finite reference loss.')
  _finite_tree(params, 'initial meta-parameters')
  # A supplied gradient can execute independent student replays sequentially.
  evaluate = jax.jit(losses) if gradient_fn is None else losses
  # The second output is auxiliary: held-out loss never contributes gradients.
  value_and_grad = (
      jax.jit(jax.value_and_grad(losses, has_aux=True))
      if gradient_fn is None else gradient_fn
  )
  baseline = evaluate(params)
  repeated = evaluate(params)
  _finite_tree((baseline, repeated), 'fixed-loss baseline')
  replay_error = max(abs(float(a) - float(b)) for a, b in zip(baseline, repeated))
  if replay_error > loss_tolerance:
    raise RuntimeError('The fixed problem is not repeatable at the requested tolerance.')
  if expected_initial_fit_loss is not None:
    reference_error = abs(float(baseline[0]) - expected_initial_fit_loss)
    if reference_error > reference_tolerance:
      raise RuntimeError(
          f'Initial fixed-fit loss does not match the reference: error={reference_error:.3g}, '
          f'tolerance={reference_tolerance:.3g}. Restore the rate-sweep checkpoint, '
          'teacher, agent settings and fixed-problem configuration before training.'
      )
    print(f'Initial fixed-fit reference error: {reference_error:.3g}.')
  initial_params = params
  opt_state = optimizer.init(params)
  factors = sorted(set((0.0, applied_multiplier, *step_multipliers)))
  training = [dict(
      update=0, fit_loss=float(baseline[0]), heldout_loss=float(baseline[1]),
      fit_loss_before=float(baseline[0]), fit_loss_change=0.0,
      gradient_norm_before=0.0, proposed_step_norm=0.0, applied_step_norm=0.0,
      resolved_decrease=False, resolved_increase=False,
  )]
  trials = []
  for update in range(1, num_updates + 1):
    (before, heldout_before), grads = value_and_grad(params)
    _finite_tree((before, heldout_before), f'losses at update {update}')
    gradient_norm = _norm(grads)
    updates, next_opt_state = optimizer.update(grads, opt_state, params)
    proposed_norm = _norm(updates)
    if not np.isfinite((gradient_norm, proposed_norm)).all():
      raise FloatingPointError(f'Non-finite gradient/update norm at meta-update {update}.')
    committed = None
    for factor in factors:
      if factor == 0:
        candidate, fit, heldout, actual_norm, prediction = params, float(before), float(heldout_before), 0.0, 0.0
      else:
        candidate = jax.tree.map(lambda p, u: p + factor * u, params, updates)
        fit_array, heldout_array = evaluate(candidate)
        fit, heldout = float(fit_array), float(heldout_array)
        perturbation = jax.tree.map(lambda p, q: p - q, candidate, params)
        actual_norm, prediction = _norm(perturbation), _dot(grads, perturbation)
      if not np.isfinite((fit, heldout, actual_norm, prediction)).all():
        raise FloatingPointError(f'Non-finite trial at update {update}, multiplier={factor}.')
      change = fit - float(before)
      trials.append(dict(
          update=update, multiplier=factor, committed=factor == applied_multiplier,
          fit_loss_before=float(before), fit_loss=fit, fit_loss_change=change,
          heldout_loss=heldout, heldout_loss_change=heldout - float(heldout_before),
          proposed_step_norm=proposed_norm, actual_step_norm=actual_norm,
          predicted_fit_change=prediction,
          resolved_decrease=change < -loss_tolerance,
          resolved_increase=change > loss_tolerance,
      ))
      if factor == applied_multiplier:
        committed = candidate, fit, heldout, actual_norm
    assert committed is not None
    params, fit, heldout, applied_norm = committed
    # Keep only eta and outer optimiser history; no inner learner state escapes.
    params, opt_state = jax.tree.map(jax.lax.stop_gradient, (params, next_opt_state))
    change = fit - float(before)
    training.append(dict(
        update=update, fit_loss=fit, heldout_loss=heldout,
        fit_loss_before=float(before), fit_loss_change=change,
        gradient_norm_before=gradient_norm, proposed_step_norm=proposed_norm,
        applied_step_norm=applied_norm,
        resolved_decrease=change < -loss_tolerance,
        resolved_increase=change > loss_tolerance,
    ))
    if update % log_every == 0 or update == num_updates:
      print(
          f'fixed meta-update={update}: fit={fit:.7f}, held-out={heldout:.7f}, '
          f'actual change={change:+.3g}, applied norm={applied_norm:.3g}'
      )

  history, candidates = pd.DataFrame(training), pd.DataFrame(trials)
  fig, axes = plt.subplots(1, 3, figsize=(17, 4))
  axes[0].plot(history['update'], history.fit_loss, label='Fixed fitting states')
  axes[0].plot(history['update'], history.heldout_loss, label='Fixed held-out states')
  axes[0].set(title='Post-meta-update policy distance', xlabel='Meta-updates', ylabel='Smooth Wasserstein')
  active = history[history['update'] > 0]
  axes[1].plot(active['update'], active.gradient_norm_before, label='Meta-gradient norm')
  axes[1].plot(active['update'], active.applied_step_norm, label='Applied eta-step norm')
  axes[1].set(title='Gradient and parameter-step norms', xlabel='Meta-updates', yscale='symlog')
  for factor in factors:
    if factor > 0:
      group = candidates[candidates['multiplier'] == factor]
      suffix = ' (committed)' if factor == applied_multiplier else ''
      axes[2].plot(group['update'], group.fit_loss_change, label=f'{factor:g} x Adam{suffix}')
  axes[2].axhline(0, color='black', linewidth=0.8)
  axes[2].axhspan(-loss_tolerance, loss_tolerance, color='grey', alpha=0.15)
  axes[2].set(title='Trial changes from the same pre-update eta', xlabel='Meta-updates', ylabel='Actual fitting-loss change', yscale='symlog')
  for ax in axes:
    ax.legend(fontsize=8)
  fig.tight_layout()
  return FixedMetaTrainingResult(
      history, candidates, initial_params, params, opt_state, replay_error,
      applied_multiplier, fig,
  )


def fit_fixed_meta(
    teacher: PPOResult, agent_settings: config_dict.ConfigDict,
    config: MetaTrainingConfig, *,
    meta_params: types.MetaParams | None = None,
    heldout_seed: int = 913,
    step_multipliers: tuple[float, ...] = (0.01, 0.1, 1.0),
    applied_multiplier: float = 1.0,
    loss_tolerance: float = 1e-7,
    log_every: int = 10,
    expected_initial_fit_loss: float | None = None,
    reference_tolerance: float = 1e-6,
) -> FixedMetaTrainingResult:
  """Optimise eta repeatedly, resetting the same inner learner on every replay."""
  if heldout_seed in config.seeds:
    raise ValueError('Use a held-out seed distinct from the fitting student seed.')
  trainer, params, learner, train, fitting, learn_key = _fixed_snapshot(
      teacher, agent_settings, config, meta_params
  )
  reset_key, actor_key, collect_key = jax.random.split(
      jax.random.PRNGKey(heldout_seed), 3
  )
  env_state, timestep = trainer.student.env.reset(reset_key)
  actor = trainer.agent.initial_actor_state(actor_key)
  heldout, _, _, _ = trainer.collect_validation(
      learner.params, actor, timestep, env_state, collect_key
  )
  heldout = jax.tree.map(jax.lax.stop_gradient, heldout)
  if not bool((heldout.discounts[:-1] > 0).any()):
    raise RuntimeError('The held-out snapshot contains no valid policy positions.')
  targets = trainer.student.teacher_samples(heldout.observations['observation'][:-1])

  def losses(eta):
    fit, (updated, _) = trainer.outer_loss(eta, learner, train, fitting, learn_key)
    _, (_, metrics) = trainer.student.policy_loss(
        updated.params, heldout.first_state(time_axis=0), heldout, targets
    )
    return fit, metrics['pi_loss']

  print(
      f'Fixed problem: {config.num_inner_steps} inner updates, '
      f'{config.rollout_len} rollout steps, {config.num_envs} environments, '
      f'{trainer.agent.settings.update_rule.num_samples} samples.'
  )
  print(
      f'Committing {applied_multiplier:g} x the current Adam update every iteration; '
      'other multipliers are diagnostics only. No line search or held-out selection.'
  )
  result = _optimise_fixed_loss(
      params, losses, trainer.meta_optimizer, config.num_updates,
      step_multipliers, applied_multiplier, loss_tolerance, log_every,
      expected_initial_fit_loss=expected_initial_fit_loss,
      reference_tolerance=reference_tolerance,
  )
  print(f'Baseline replay error: {result.replay_error:.3g}. Inner states/data were reset for every trial.')
  return result


def probe_fixed_meta_updates(
    teacher: PPOResult, agent_settings: config_dict.ConfigDict,
    config: MetaTrainingConfig, result: FixedMetaTrainingResult, *,
    step_norms: tuple[float, ...] = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1),
    loss_tolerance: float = 1e-7,
    reference_tolerance: float = 1e-6,
) -> MetaUpdateProbeResult:
  """Check trained eta and its next Adam proposal on the original fixed problem."""
  return probe_meta_updates(
      teacher, agent_settings, config, meta_params=result.final_meta_params,
      meta_opt_state=result.final_meta_opt_state,
      expected_loss=float(result.training.fit_loss.iloc[-1]),
      step_norms=step_norms, loss_tolerance=loss_tolerance,
      reference_tolerance=reference_tolerance,
  )


def save_fixed_meta_results(result: FixedMetaTrainingResult, output_dir: Path) -> None:
  output_dir.mkdir(parents=True, exist_ok=True)
  result.training.to_csv(output_dir / 'training.csv', index=False)
  result.step_trials.to_csv(output_dir / 'step_trials.csv', index=False)
  pd.DataFrame([dict(
      replay_error=result.replay_error, applied_multiplier=result.applied_multiplier,
      initial_fit_loss=result.training.fit_loss.iloc[0],
      final_fit_loss=result.training.fit_loss.iloc[-1],
      initial_heldout_loss=result.training.heldout_loss.iloc[0],
      final_heldout_loss=result.training.heldout_loss.iloc[-1],
  )]).to_csv(output_dir / 'summary.csv', index=False)
  result.figure.savefig(output_dir / 'learning_curves.png', dpi=150)
  print(f'Fixed-problem meta-training results saved to {output_dir}')
