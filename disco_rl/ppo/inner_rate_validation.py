"""Compare inner learning rates using one immutable agent/data snapshot."""

from dataclasses import dataclass
from pathlib import Path

import jax
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from ml_collections import config_dict
import numpy as np
import pandas as pd

from disco_rl import types
from disco_rl.ppo.meta_training import MetaTrainingConfig
from disco_rl.ppo.meta_training import MetaTrainingTrainer
from disco_rl.ppo.meta_update_validation import _finite_tree
from disco_rl.ppo.meta_update_validation import _fixed_snapshot
from disco_rl.ppo.meta_update_validation import _measure_updates
from disco_rl.ppo.meta_update_validation import _norm
from disco_rl.ppo.training import PPOResult


@dataclass
class InnerRateValidationResult:
  summary: pd.DataFrame
  trials: pd.DataFrame
  slopes: pd.DataFrame
  direction_status: pd.DataFrame
  figure: Figure


def probe_inner_learning_rates(
    teacher: PPOResult, agent_settings: config_dict.ConfigDict,
    config: MetaTrainingConfig, *,
    meta_params: types.MetaParams | None = None,
    learning_rates: tuple[float, ...] = (0.05, 0.005, 0.0005),
    step_norms: tuple[float, ...] = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1),
    expected_baseline_loss: float | None = None,
    loss_tolerance: float = 1e-7,
    reference_tolerance: float = 1e-6,
) -> InnerRateValidationResult:
  """Change only the inner rate; neither eta nor any optimiser state is trained."""
  original_rate = float(agent_settings.learning_rate)
  if (
      not learning_rates or len(set(learning_rates)) != len(learning_rates)
      or any(not np.isfinite(r) or r <= 0 for r in (*learning_rates, original_rate))
      or not step_norms or len(set(step_norms)) != len(step_norms)
      or any(not np.isfinite(s) or s <= 0 for s in step_norms)
      or not np.isfinite(loss_tolerance) or loss_tolerance <= 0
      or not np.isfinite(reference_tolerance) or reference_tolerance <= 0
      or (expected_baseline_loss is not None and not np.isfinite(expected_baseline_loss))
  ):
    raise ValueError('Use distinct positive finite rates/steps, positive tolerances and a finite reference loss.')
  rates = tuple(dict.fromkeys((original_rate, *learning_rates)))
  base, params, learner, train, fitting, learn_key = _fixed_snapshot(
      teacher, agent_settings, config, meta_params
  )
  targets = base.student.teacher_samples(fitting.observations['observation'][:-1])

  def initial_policy_loss():
    _, (_, metrics) = base.student.policy_loss(
        learner.params, fitting.first_state(time_axis=0), fitting, targets
    )
    return metrics['pi_loss']

  before_inner = float(jax.jit(initial_policy_loss)())
  _finite_tree(before_inner, 'pre-inner policy loss')
  print(
      f'One snapshot, {config.num_inner_steps} inner updates, '
      f'{config.rollout_len} rollout steps, {config.num_envs} environments, '
      f'{base.agent.settings.update_rule.num_samples} samples. '
      f'Outer rate stays {config.meta_learning_rate:g}; no meta-updates committed.'
  )
  summary, trials, slopes, statuses = [], [], [], []
  for rate in rates:
    settings = config_dict.ConfigDict(agent_settings.to_dict())
    settings.learning_rate = rate
    trainer = base if rate == original_rate else MetaTrainingTrainer(teacher, settings, config)

    def objective(eta):
      return trainer.outer_loss(eta, learner, train, fitting, learn_key)

    evaluate = jax.jit(lambda eta: objective(eta)[0])
    baseline = float(evaluate(params))
    _finite_tree(baseline, f'baseline at inner rate {rate:g}')
    reference_error = None
    if rate == original_rate and expected_baseline_loss is not None:
      reference_error = abs(baseline - expected_baseline_loss)
      if reference_error > reference_tolerance:
        raise RuntimeError(
            f'Original-rate checkpoint replay does not match: error={reference_error:.3g}. '
            'Restore the original teacher, agent settings, seeds and fixed-problem configuration.'
        )
    (gradient_loss, (updated, logs)), grads = jax.jit(
        jax.value_and_grad(objective, has_aux=True)
    )(params)
    _finite_tree((updated, logs, grads), f'inner replay at rate {rate:g}')
    if abs(float(gradient_loss) - baseline) > reference_tolerance:
      raise RuntimeError(f'Gradient/scalar loss evaluations disagree at inner rate {rate:g}.')
    probe = _measure_updates(
        evaluate, params, grads, trainer.meta_optimizer, step_norms, loss_tolerance
    )
    agent_change = jax.tree.map(lambda p, q: p - q, updated.params, learner.params)
    descent = probe.trials[
        (probe.trials.direction == 'negative_gradient') & (probe.trials.sign == 1)
    ]
    summary.append(dict(
        inner_learning_rate=rate, original_rate=rate == original_rate,
        outer_learning_rate=config.meta_learning_rate,
        num_inner_steps=config.num_inner_steps, rollout_len=config.rollout_len,
        num_envs=config.num_envs, num_samples=trainer.agent.settings.update_rule.num_samples,
        policy_loss_before_inner=before_inner, policy_loss_after_inner=probe.baseline_loss,
        inner_policy_loss_change=probe.baseline_loss - before_inner,
        meta_gradient_norm=probe.gradient_norm,
        gradient_nonzero_count=probe.gradient_nonzero_count,
        gradient_max_abs=probe.gradient_max_abs,
        agent_parameter_change_norm=_norm(agent_change),
        mean_inner_agent_gradient_norm=float(logs['agent_gradient_norm']),
        mean_inner_agent_update_norm=float(logs['agent_update_norm']),
        negative_gradient_available=not descent.empty,
        best_negative_gradient_loss_change=(
            float(descent.actual_loss_change.min()) if not descent.empty else np.nan
        ),
        resolved_negative_gradient_decreases=int(descent.resolved_decrease.sum()),
        fresh_adam_proposal_norm=probe.fresh_adam_update_norm,
        replay_error=probe.replay_error, reference_loss_error=reference_error,
    ))
    for collection, table in (
        (trials, probe.trials), (slopes, probe.slopes), (statuses, probe.direction_status),
    ):
      collection.append(table.assign(inner_learning_rate=rate))
    plt.close(probe.figure)
    print(
        f'Inner rate={rate:g}: post-inner W={probe.baseline_loss:.7f}, '
        f'meta-gradient={probe.gradient_norm:.6g}, '
        f'nonzero entries={probe.gradient_nonzero_count}, '
        f'resolved negative-gradient improvements={int(descent.resolved_decrease.sum())}.'
    )

  table = pd.DataFrame(summary)
  all_trials, all_slopes = pd.concat(trials, ignore_index=True), pd.concat(slopes, ignore_index=True)
  fig, axes = plt.subplots(1, 3, figsize=(17, 4))
  ordered = table.sort_values('inner_learning_rate')
  axes[0].plot(ordered.inner_learning_rate, ordered.meta_gradient_norm, marker='o')
  axes[0].set(
      title='Meta-gradient sensitivity', xlabel='Inner learning rate',
      ylabel='Meta-gradient L2 norm', xscale='log', yscale='symlog',
  )
  axes[1].plot(ordered.inner_learning_rate, ordered.policy_loss_after_inner, marker='o', label='After inner updates')
  axes[1].axhline(before_inner, color='black', linestyle=':', label='Before inner updates')
  axes[1].set(title='Same rule, same fitting observations', xlabel='Inner learning rate', ylabel='Policy distance to PPO', xscale='log')
  axes[1].legend(fontsize=7)
  for rate in rates:
    forward = all_trials[(all_trials.inner_learning_rate == rate) & (all_trials.sign == 1)]
    for direction, group in forward.groupby('direction'):
      axes[2].plot(
          group.requested_step_norm, group.actual_loss_change, marker='o',
          label=f'{rate:g}: {direction}',
      )
  axes[2].axhline(0, color='black', linewidth=0.8)
  axes[2].axhspan(-loss_tolerance, loss_tolerance, color='grey', alpha=0.15)
  axes[2].set(
      title='Positive-direction trials (opposite controls in CSV)',
      xlabel='Absolute meta-parameter step norm', ylabel='Measured fitting-loss change',
      xscale='log', yscale='symlog',
  )
  axes[2].legend(fontsize=6)
  fig.tight_layout()
  return InnerRateValidationResult(
      table, all_trials, all_slopes, pd.concat(statuses, ignore_index=True), fig
  )


def save_inner_rate_validation(result: InnerRateValidationResult, output_dir: Path) -> None:
  output_dir.mkdir(parents=True, exist_ok=True)
  for name in ('summary', 'trials', 'slopes', 'direction_status'):
    getattr(result, name).to_csv(output_dir / f'{name}.csv', index=False)
  result.figure.savefig(output_dir / 'learning_rate_comparison.png', dpi=150)
  print(f'Inner-rate validation saved to {output_dir}')
