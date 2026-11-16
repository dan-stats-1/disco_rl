"""Diagnose student-gradient conflict without training or selecting a new rule."""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import haiku as hk
import jax
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from ml_collections import config_dict
import numpy as np
import pandas as pd

from disco_rl.ppo.fixed_student_bank import FixedStudentBankResult
from disco_rl.ppo.meta_training import MetaTrainingConfig
from disco_rl.ppo.meta_update_validation import _collect_fixed_snapshot
from disco_rl.ppo.meta_update_validation import _dot
from disco_rl.ppo.meta_update_validation import _finite_tree
from disco_rl.ppo.meta_update_validation import _fixed_snapshot
from disco_rl.ppo.meta_update_validation import _norm
from disco_rl.ppo.training import PPOResult


@dataclass
class StudentGradientAlignmentResult:
  students: pd.DataFrame
  pairs: pd.DataFrame
  directions: pd.DataFrame
  trials: pd.DataFrame
  summary: pd.DataFrame
  configuration: pd.DataFrame
  figure: Figure


def _measure_alignment(
    params: hk.Params, training_seeds: tuple[int, ...],
    validation_seeds: tuple[int, ...],
    evaluate: Callable[[hk.Params, int], jax.Array],
    differentiate: Callable[[hk.Params, int], tuple[jax.Array, hk.Params]],
    step_norms: tuple[float, ...], *, checkpoint: str,
    reference_losses: dict[int, float], reference_tolerance: float,
    loss_tolerance: float,
):
  if (
      not step_norms or len(set(step_norms)) != len(step_norms)
      or any(not np.isfinite(s) or s <= 0 for s in step_norms)
      or not np.isfinite(reference_tolerance) or reference_tolerance <= 0
      or not np.isfinite(loss_tolerance) or loss_tolerance <= 0
  ):
    raise ValueError('Use distinct positive finite step norms and positive finite tolerances.')
  _finite_tree((params, reference_losses), 'alignment parameters/references')
  baseline = {}
  grads = {}
  for seed in (*training_seeds, *validation_seeds):
    baseline[seed] = float(evaluate(params, seed))
    _finite_tree(baseline[seed], f'{checkpoint} student {seed} baseline')
    if abs(baseline[seed] - reference_losses[seed]) > reference_tolerance:
      raise RuntimeError(
          f'{checkpoint} student {seed} does not match the saved bank loss; '
          'restore the completed bank teacher, rule, settings and configuration.'
      )
  for seed in training_seeds:
    value, gradient = differentiate(params, seed)
    _finite_tree((value, gradient), f'{checkpoint} student {seed} gradient')
    if abs(float(value) - baseline[seed]) > reference_tolerance:
      raise RuntimeError(f'{checkpoint} student {seed} forward/gradient replay mismatch.')
    grads[seed] = gradient
    jax.block_until_ready(gradient)
  norms = {seed: _norm(g) for seed, g in grads.items()}

  def mean(trees):
    return jax.tree.map(lambda *xs: sum(xs) / len(xs), *trees)

  raw_mean = mean(list(grads.values()))
  # A zero-gradient student contributes zero, not an arbitrary replacement direction.
  balanced_mean = mean([
      jax.tree.map(lambda g: g / norms[s], grads[s]) if norms[s] > 0 else grads[s]
      for s in training_seeds
  ])
  raw_norm = _norm(raw_mean)

  def cosine(a, b):
    denominator = _norm(a) * _norm(b)
    return _dot(a, b) / denominator if denominator > 0 else np.nan

  norm_sum = sum(norms.values())
  students = [
      dict(
          checkpoint=checkpoint, split='training' if seed in grads else 'validation',
          seed=seed, policy_distance=baseline[seed],
          reference_loss=reference_losses[seed],
          reference_error=abs(baseline[seed] - reference_losses[seed]),
          gradient_norm=norms.get(seed, np.nan),
          fraction_of_gradient_norm_sum=(
              norms[seed] / norm_sum if seed in grads and norm_sum > 0 else np.nan
          ),
          cosine_to_raw_mean=cosine(grads[seed], raw_mean) if seed in grads else np.nan,
      )
      for seed in (*training_seeds, *validation_seeds)
  ]
  pairs = [
      dict(
          checkpoint=checkpoint, left_seed=left, right_seed=right,
          gradient_dot=_dot(grads[left], grads[right]),
          cosine=cosine(grads[left], grads[right]),
      )
      for i, left in enumerate(training_seeds) for right in training_seeds[i + 1:]
  ]
  directions, trials = [], []
  for name, vector in (('raw_mean', raw_mean), ('norm_balanced_mean', balanced_mean)):
    norm = _norm(vector)
    _finite_tree((vector, norm), f'{checkpoint} {name} direction')
    directions.append(dict(
        checkpoint=checkpoint, direction=name, unnormalised_norm=norm,
        tested=norm > 0, status='available' if norm > 0 else 'unavailable_zero_norm',
        cosine_to_raw_mean=cosine(vector, raw_mean) if raw_norm > 0 else np.nan,
    ))
    if norm == 0:
      continue
    for step in step_norms:
      for sign in (-1, 1):
        candidate = jax.tree.map(lambda p, g: p + sign * step * g / norm, params, vector)
        delta = jax.tree.map(lambda p, q: p - q, candidate, params)
        actual_norm = _norm(delta)
        _finite_tree((candidate, delta, actual_norm), f'{checkpoint} {name} perturbation')
        for seed in (*training_seeds, *validation_seeds):
          loss = float(evaluate(candidate, seed))
          _finite_tree(loss, f'{checkpoint} {name} student {seed} trial')
          change = loss - baseline[seed]
          prediction = _dot(grads[seed], delta) if seed in grads else np.nan
          if seed in grads:
            _finite_tree(prediction, f'{checkpoint} {name} student {seed} prediction')
          trials.append(dict(
              checkpoint=checkpoint, direction=name, sign=sign,
              requested_step_norm=step, actual_step_norm=actual_norm,
              split='training' if seed in grads else 'validation', seed=seed,
              baseline_loss=baseline[seed], loss=loss, loss_change=change,
              predicted_change=prediction,
              resolved_decrease=change < -loss_tolerance,
              resolved_increase=change > loss_tolerance,
          ))
  return students, pairs, directions, trials


def probe_student_gradient_alignment(
    teacher: PPOResult, agent_settings: config_dict.ConfigDict,
    config: MetaTrainingConfig, bank: FixedStudentBankResult, *,
    step_norms: tuple[float, ...] = (1e-4, 1e-3, 1e-2),
    reference_tolerance: float = 1e-6, loss_tolerance: float = 1e-7,
) -> StudentGradientAlignmentResult:
  """Replay initial/final bank problems; only training students supply gradients."""
  if len(bank.configuration) != 1:
    raise ValueError('Expected exactly one completed student-bank configuration.')
  saved = bank.configuration.iloc[0]
  current = dict(
      num_updates=config.num_updates, num_inner_steps=config.num_inner_steps,
      rollout_len=config.rollout_len, num_envs=config.num_envs,
      num_samples=agent_settings.update_rule.num_samples,
      inner_learning_rate=agent_settings.learning_rate,
      outer_learning_rate=config.meta_learning_rate,
      use_auxiliary_losses=config.use_auxiliary_losses,
  )
  for name, value in current.items():
    if saved[name] != value:
      raise RuntimeError(f'{name} changed since bank fitting; restore the completed bank settings.')
  seeds = {
      split: tuple(int(s) for s in bank.student_history.loc[
          bank.student_history.split == split, 'seed'
      ].unique())
      for split in ('training', 'validation')
  }
  training, validation = seeds['training'], seeds['validation']
  if (
      len(training) < 2 or not validation or set(training) & set(validation)
      or config.seeds != training[:1]
      or repr(training) != saved.training_seeds
      or repr(validation) != saved.validation_seeds
  ):
    raise ValueError('Student history must match the completed disjoint bank splits.')
  trainer, initial, learner, train, probe, key = _fixed_snapshot(
      teacher, agent_settings, config, bank.optimisation.initial_meta_params
  )
  problems = {training[0]: (learner, train, probe, key)}
  for seed in (*training[1:], *validation):
    problems[seed] = _collect_fixed_snapshot(trainer, seed)

  def objective(eta, student, rollouts, observations, learn_key):
    return trainer.outer_loss(eta, student, rollouts, observations, learn_key)[0]

  forward = jax.jit(objective)
  backward = jax.jit(jax.value_and_grad(objective))
  tables = [[], [], [], []]
  for name, update, params in (
      ('initial', 0, initial),
      ('final', config.num_updates, bank.optimisation.final_meta_params),
  ):
    reference = bank.student_history[bank.student_history['update'] == update]
    if len(reference) != len(problems) or reference.seed.duplicated().any():
      raise ValueError(f'Expected one recorded loss per bank student at update {update}.')
    measured = _measure_alignment(
        params, training, validation,
        lambda eta, seed: forward(eta, *problems[seed]),
        lambda eta, seed: backward(eta, *problems[seed]),
        step_norms, checkpoint=name,
        reference_losses=reference.set_index('seed').policy_distance.to_dict(),
        reference_tolerance=reference_tolerance, loss_tolerance=loss_tolerance,
    )
    for table, rows in zip(tables, measured):
      table.extend(rows)
    print(f'Completed {name} bank gradient alignment; no parameters or optimiser states changed.')
  students, pairs, directions, trials = (pd.DataFrame(t) for t in tables)
  if trials.empty:
    summary = pd.DataFrame(columns=[
        'checkpoint', 'direction', 'sign', 'requested_step_norm', 'split',
        'baseline_loss_mean', 'loss_mean', 'loss_change_mean',
        'predicted_change_mean', 'students_improved', 'students_worsened',
    ])
    print('WARNING: All tested directions have zero norm; inspect direction_status.csv.')
  else:
    summary = trials.groupby(
        ['checkpoint', 'direction', 'sign', 'requested_step_norm', 'split'], as_index=False,
    ).agg(
        baseline_loss_mean=('baseline_loss', 'mean'), loss_mean=('loss', 'mean'),
        loss_change_mean=('loss_change', 'mean'), predicted_change_mean=('predicted_change', 'mean'),
        students_improved=('resolved_decrease', 'sum'), students_worsened=('resolved_increase', 'sum'),
    )
  fig, axes = plt.subplots(2, 3, figsize=(18, 8))
  for row, name in enumerate(('initial', 'final')):
    group = students[(students.checkpoint == name) & (students.split == 'training')]
    axes[row, 0].bar(group.seed.astype(str), group.gradient_norm)
    axes[row, 0].set(title=f'{name}: training-gradient norms', yscale='symlog', xlabel='Student seed')
    group = pairs[pairs.checkpoint == name]
    labels = group.left_seed.astype(str) + '/' + group.right_seed.astype(str)
    axes[row, 1].bar(labels, group.cosine)
    axes[row, 1].axhline(0, color='black', linewidth=0.8)
    axes[row, 1].set(title=f'{name}: pairwise gradient cosine', ylim=(-1.05, 1.05))
    axes[row, 1].tick_params(axis='x', labelrotation=45)
    for direction in ('raw_mean', 'norm_balanced_mean'):
      for split in ('training', 'validation'):
        group = summary[
            (summary.checkpoint == name) & (summary.direction == direction)
            & (summary.sign == -1) & (summary.split == split)
        ]
        if not group.empty:
          axes[row, 2].plot(
              group.requested_step_norm, group.loss_change_mean, marker='o',
              label=f'{direction}: {split}',
          )
    axes[row, 2].axhline(0, color='black', linewidth=0.8)
    axes[row, 2].set(
        title=f'{name}: negative-direction measured change',
        xlabel='Requested absolute eta-step L2 norm', xscale='log',
        ylabel='Mean policy-distance change (negative is better)',
    )
    if axes[row, 2].lines[1:]:
      axes[row, 2].legend(fontsize=7)
  fig.tight_layout()
  configuration = bank.configuration.copy()
  configuration['step_norms'] = repr(step_norms)
  configuration['reference_tolerance'] = reference_tolerance
  configuration['loss_tolerance'] = loss_tolerance
  return StudentGradientAlignmentResult(
      students, pairs, directions, trials, summary, configuration, fig
  )


def save_student_gradient_alignment(
    result: StudentGradientAlignmentResult, output_dir: Path,
) -> None:
  output_dir.mkdir(parents=True, exist_ok=True)
  for name, table in (
      ('students', result.students), ('gradient_pairs', result.pairs),
      ('direction_status', result.directions), ('trials', result.trials),
      ('summary', result.summary), ('configuration', result.configuration),
  ):
    table.to_csv(output_dir / f'{name}.csv', index=False)
  result.figure.savefig(output_dir / 'gradient_alignment.png', dpi=150)
