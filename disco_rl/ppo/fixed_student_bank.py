"""Fit one rule to a fixed bank of reset students with sequential meta-gradients."""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import haiku as hk
import jax
import jax.numpy as jnp
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from ml_collections import config_dict
import numpy as np
import pandas as pd

from disco_rl import types
from disco_rl.ppo.fixed_meta_training import FixedMetaTrainingResult
from disco_rl.ppo.fixed_meta_training import _optimise_fixed_loss
from disco_rl.ppo.fixed_meta_training import save_fixed_meta_results
from disco_rl.ppo.meta_training import MetaTrainingConfig
from disco_rl.ppo.meta_update_validation import _collect_fixed_snapshot
from disco_rl.ppo.meta_update_validation import _finite_tree
from disco_rl.ppo.meta_update_validation import _fixed_snapshot
from disco_rl.ppo.meta_update_validation import _norm
from disco_rl.ppo.training import PPOResult


@dataclass
class FixedStudentBankResult:
  optimisation: FixedMetaTrainingResult
  student_history: pd.DataFrame
  configuration: pd.DataFrame
  student_figure: Figure


def _sequential_objectives(
    training_seeds: tuple[int, ...], validation_seeds: tuple[int, ...],
    evaluate: Callable[[hk.Params, int], jax.Array],
    differentiate: Callable[[hk.Params, int], tuple[jax.Array, hk.Params]],
):
  rows = []
  completed_updates = 0

  def losses(params):
    values = []
    for seeds in (training_seeds, validation_seeds):
      group = [float(evaluate(params, seed)) for seed in seeds]
      _finite_tree(group, 'student-bank losses')
      values.append(jnp.asarray(np.mean(group), dtype=jnp.float32))
    return values[0], values[1]

  def gradient(params):
    nonlocal completed_updates
    total = jax.tree.map(jnp.zeros_like, params)
    fits = []
    for seed in training_seeds:
      value, grads = differentiate(params, seed)
      _finite_tree((value, grads), f'training student {seed}')
      fits.append(float(value))
      rows.append(dict(
          update=completed_updates, split='training', seed=seed,
          policy_distance=float(value), gradient_norm_for_next_update=_norm(grads),
      ))
      total = jax.tree.map(lambda a, b: a + b, total, grads)
      # Finish each accumulation before launching the next student's backprop.
      jax.block_until_ready(total)
    heldout = []
    for seed in validation_seeds:
      value = float(evaluate(params, seed))
      _finite_tree(value, f'validation student {seed}')
      heldout.append(value)
      rows.append(dict(
          update=completed_updates, split='validation', seed=seed,
          policy_distance=value, gradient_norm_for_next_update=np.nan,
      ))
    completed_updates += 1
    average = jax.tree.map(lambda g: g / len(training_seeds), total)
    return (
        jnp.asarray(np.mean(fits), dtype=jnp.float32),
        jnp.asarray(np.mean(heldout), dtype=jnp.float32),
    ), average

  def record_final(params):
    for split, seeds in (('training', training_seeds), ('validation', validation_seeds)):
      for seed in seeds:
        value = float(evaluate(params, seed))
        _finite_tree(value, f'final {split} student {seed}')
        rows.append(dict(
            update=completed_updates, split=split, seed=seed,
            policy_distance=value, gradient_norm_for_next_update=np.nan,
        ))

  return losses, gradient, rows, record_final


def fit_fixed_student_bank(
    teacher: PPOResult, agent_settings: config_dict.ConfigDict,
    config: MetaTrainingConfig, *, meta_params: types.MetaParams,
    training_seeds: tuple[int, ...] = (811, 815, 816, 817),
    validation_seeds: tuple[int, ...] = (812, 813, 814),
    reserved_test_seeds: tuple[int, ...] = (831, 832, 833),
    step_multipliers: tuple[float, ...] = (0.01, 0.1, 1.0),
    applied_multiplier: float = 1.0,
    expected_original_fit_loss: float | None = None,
    loss_tolerance: float = 1e-7, reference_tolerance: float = 1e-6,
    log_every: int = 10,
) -> FixedStudentBankResult:
  """Average training gradients; validation is forward-only and tests untouched."""
  all_seeds = (*training_seeds, *validation_seeds, *reserved_test_seeds)
  if (
      len(training_seeds) < 2 or not validation_seeds or not reserved_test_seeds
      or len(set(all_seeds)) != len(all_seeds)
      or any(not isinstance(s, int) or isinstance(s, bool) or s < 0 or s >= 2**32 for s in all_seeds)
      or config.seeds != training_seeds[:1]
      or not np.isfinite(reference_tolerance) or reference_tolerance <= 0
      or (expected_original_fit_loss is not None and not np.isfinite(expected_original_fit_loss))
  ):
    raise ValueError('Use disjoint valid training/validation/reserved seeds, at least two training students, the original student first, and a finite reference/tolerance.')
  trainer, params, learner, train, probe, key = _fixed_snapshot(
      teacher, agent_settings, config, meta_params
  )
  problems = {training_seeds[0]: (learner, train, probe, key)}
  for seed in (*training_seeds[1:], *validation_seeds):
    problems[seed] = _collect_fixed_snapshot(trainer, seed)

  def objective(eta, student, rollouts, observations, learn_key):
    return trainer.outer_loss(eta, student, rollouts, observations, learn_key)[0]

  forward = jax.jit(objective)
  backward = jax.jit(jax.value_and_grad(objective))

  def evaluate(eta, seed):
    return forward(eta, *problems[seed])

  def differentiate(eta, seed):
    return backward(eta, *problems[seed])

  original_loss = float(evaluate(params, training_seeds[0]))
  _finite_tree(original_loss, 'original student baseline')
  if expected_original_fit_loss is not None:
    reference_error = abs(original_loss - expected_original_fit_loss)
    if reference_error > reference_tolerance:
      raise RuntimeError(
          f'Original-student pre-fit loss does not match: error={reference_error:.3g}. '
          'Use the pre-single-student-fit rule and unchanged settings/configuration.'
      )
  else:
    reference_error = np.nan
  print(
      f'Fixed student bank: training={training_seeds}, validation={validation_seeds}. '
      f'Reserved tests={reserved_test_seeds} will not be collected or evaluated.\n'
      f'{config.num_inner_steps} inner updates; inner rate={agent_settings.learning_rate:g}; '
      f'outer rate={config.meta_learning_rate:g}. Equal-weight gradients are '
      'accumulated sequentially before one outer update.'
  )
  losses, gradient, rows, record_final = _sequential_objectives(
      training_seeds, validation_seeds, evaluate, differentiate
  )
  fit = _optimise_fixed_loss(
      params, losses, trainer.meta_optimizer, config.num_updates,
      step_multipliers, applied_multiplier, loss_tolerance, log_every,
      gradient_fn=gradient,
  )
  record_final(fit.final_meta_params)
  history = pd.DataFrame(rows)
  first_axis = fit.figure.axes[0]
  first_axis.lines[0].set_label('Mean training-student distance')
  first_axis.lines[1].set_label('Mean validation-student distance')
  first_axis.legend(fontsize=8)
  student_fig, axes = plt.subplots(1, 2, figsize=(13, 4))
  for axis, split in zip(axes, ('training', 'validation')):
    for seed, group in history[history.split == split].groupby('seed'):
      axis.plot(group['update'], group.policy_distance, label=f'student {seed}')
    axis.set(title=f'Fixed {split} students', xlabel='Meta-updates', ylabel='PPO policy distance')
    axis.legend(fontsize=8)
  student_fig.tight_layout()
  configuration = pd.DataFrame([dict(
      training_seeds=repr(training_seeds), validation_seeds=repr(validation_seeds),
      reserved_test_seeds=repr(reserved_test_seeds),
      num_updates=config.num_updates, num_inner_steps=config.num_inner_steps,
      rollout_len=config.rollout_len, num_envs=config.num_envs,
      num_samples=trainer.agent.settings.update_rule.num_samples,
      inner_learning_rate=agent_settings.learning_rate,
      outer_learning_rate=config.meta_learning_rate,
      use_auxiliary_losses=config.use_auxiliary_losses,
      applied_multiplier=applied_multiplier, original_initial_fit_loss=original_loss,
      original_reference_error=reference_error,
  )])
  return FixedStudentBankResult(fit, history, configuration, student_fig)


def save_fixed_student_bank(result: FixedStudentBankResult, output_dir: Path) -> None:
  save_fixed_meta_results(result.optimisation, output_dir)
  result.student_history.to_csv(output_dir / 'students.csv', index=False)
  result.configuration.to_csv(output_dir / 'configuration.csv', index=False)
  result.student_figure.savefig(output_dir / 'student_curves.png', dpi=150)
