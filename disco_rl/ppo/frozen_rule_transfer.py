"""Paired frozen-rule comparisons on new experience and student initialisations."""

from dataclasses import dataclass
from pathlib import Path

import jax
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from ml_collections import config_dict
import pandas as pd

from disco_rl import types
from disco_rl import utils
from disco_rl.ppo.meta_training import MetaTrainingConfig
from disco_rl.ppo.meta_training import MetaTrainingTrainer
from disco_rl.ppo.meta_training import _initial_student
from disco_rl.ppo.meta_update_validation import _finite_tree
from disco_rl.ppo.meta_update_validation import _norm
from disco_rl.ppo.training import PPOResult


@dataclass
class FrozenRuleTransferResult:
  evaluation: pd.DataFrame
  paired: pd.DataFrame
  summary: pd.DataFrame
  configuration: pd.DataFrame
  figure: Figure


def _collect_case(trainer, learner, experience_seed):
  reset, actor_key, collect_key, learn_key, probe_reset, probe_actor, probe_key = (
      jax.random.split(jax.random.PRNGKey(experience_seed), 7)
  )
  env_state, timestep = trainer.student.env.reset(reset)
  actor = trainer.agent.initial_actor_state(actor_key)
  rollouts = []
  for _ in range(trainer.config.num_inner_steps):
    collect_key, key = jax.random.split(collect_key)
    rollout, actor, timestep, env_state = trainer.collect(
        learner.params, actor, timestep, env_state, key
    )
    rollouts.append(rollout)
  env_state, timestep = trainer.student.env.reset(probe_reset)
  actor = trainer.agent.initial_actor_state(probe_actor)
  probe, _, _, _ = trainer.collect_validation(
      learner.params, actor, timestep, env_state, probe_key
  )
  if not bool((probe.discounts[:-1] > 0).any()):
    raise RuntimeError('No valid policy positions in the transfer probe observations.')
  return jax.tree.map(
      jax.lax.stop_gradient, (utils.tree_stack(rollouts), probe, learn_key)
  )


def compare_frozen_rules(
    teacher: PPOResult, agent_settings: config_dict.ConfigDict,
    config: MetaTrainingConfig, starting_meta_params: types.MetaParams,
    learned_meta_params: types.MetaParams, *,
    fresh_seeds: tuple[int, ...] = (812, 813, 814),
    experience_seeds: tuple[int, ...] = (921, 922, 923),
    evaluation_seed: int = 941,
) -> FrozenRuleTransferResult:
  """Compare rules with paired data/RNG and exactly one K-update teaching block."""
  if (
      len(config.seeds) != 1 or not fresh_seeds
      or len(fresh_seeds) != len(experience_seeds)
      or len(set(fresh_seeds)) != len(fresh_seeds)
      or len(set(experience_seeds)) != len(experience_seeds)
      or set(fresh_seeds) & set(config.seeds)
      or set(experience_seeds) & set(config.seeds)
      or any(
          not isinstance(s, int) or isinstance(s, bool) or s < 0 or s >= 2**32
          for s in (*config.seeds, *fresh_seeds, *experience_seeds, evaluation_seed)
      )
  ):
    raise ValueError('Use one original student seed, equally sized unique fresh/experience seed banks, and valid distinct-from-training seeds.')
  trainer = MetaTrainingTrainer(teacher, agent_settings, config)
  template, _ = trainer.agent.update_rule.init_params(jax.random.PRNGKey(config.meta_seed))
  rules = {}
  for name, params in (
      ('starting_rule', starting_meta_params), ('learned_rule', learned_meta_params),
  ):
    if (
        jax.tree.structure(params) != jax.tree.structure(template)
        or any(a.shape != b.shape for a, b in zip(jax.tree.leaves(params), jax.tree.leaves(template)))
    ):
      raise ValueError(f'{name} does not match the configured update-rule architecture.')
    _finite_tree(params, name)
    rules[name] = jax.tree.map(jax.lax.stop_gradient, params)
  replay = jax.jit(trainer.outer_loss)

  def score(params, observations):
    targets = trainer.student.teacher_samples(observations.observations['observation'][:-1])
    _, (_, metrics) = trainer.student.policy_loss(
        params, observations.first_state(time_axis=0), observations, targets
    )
    return metrics['pi_loss']

  score = jax.jit(score)
  cases = [
      ('same_initialisation_new_data', config.seeds[0], seed)
      for seed in experience_seeds
  ] + [
      ('fresh_initialisation_new_data', student, experience)
      for student, experience in zip(fresh_seeds, experience_seeds)
  ]
  rows = []
  print(
      f'Frozen-rule transfer: {len(cases)} paired cases, '
      f'{config.num_inner_steps} updates per rule, inner rate={agent_settings.learning_rate:g}, '
      f'{config.rollout_len} rollout steps, {config.num_envs} training environments. '
      'Collect all training rollouts before updating. No meta-optimisation.'
  )
  for cohort, student_seed, experience_seed in cases:
    learner, _, _, _, _ = _initial_student(trainer, student_seed)
    train, probe, learn_key = _collect_case(trainer, learner, experience_seed)
    before_distance = float(score(learner.params, probe))
    _finite_tree(before_distance, 'untrained transfer policy distance')
    eval_key = jax.random.fold_in(jax.random.PRNGKey(evaluation_seed), experience_seed)
    baseline = {}

    def evaluation_rows(params, rule, stage, distance, displacement):
      result = []
      for mean_action in (False, True):
        metrics = trainer.student.evaluation_row(
            params, eval_key, student_seed,
            0 if stage == 'before' else config.num_inner_steps, mean_action,
        )
        metrics['on_policy_pi_loss'] = metrics.pop('pi_loss')
        metrics['agent_updates'] = metrics.pop('update')
        metrics['training_env_steps'] = metrics.pop('env_steps')
        mode = metrics['mode']
        if stage == 'before':
          baseline[mode] = metrics.copy()
        result.append(dict(
            **metrics, cohort=cohort, student_seed=student_seed,
            experience_seed=experience_seed, evaluation_seed=evaluation_seed,
            rule=rule, stage=stage, probe_policy_distance=distance,
            probe_distance_change=distance - before_distance,
            catch_rate_gain=metrics['catch_rate'] - baseline[mode]['catch_rate'],
            mean_return_gain=metrics['mean_return'] - baseline[mode]['mean_return'],
            agent_parameter_change_norm=displacement,
        ))
      return result

    rows.extend(evaluation_rows(
        learner.params, 'untrained', 'before', before_distance, 0.0
    ))
    for rule, eta in rules.items():
      distance, (updated, logs) = replay(eta, learner, train, probe, learn_key)
      _finite_tree((distance, updated, logs), f'{cohort} {student_seed} {experience_seed} {rule}')
      displacement = _norm(jax.tree.map(lambda p, q: p - q, updated.params, learner.params))
      rows.extend(evaluation_rows(
          updated.params, rule, 'after', float(distance), displacement
      ))
    print(f'Completed {cohort}: student={student_seed}, experience={experience_seed}.')
  evaluation = pd.DataFrame(rows)
  keys = ['cohort', 'student_seed', 'experience_seed', 'mode']
  metrics = [
      'catch_rate', 'mean_return', 'probe_policy_distance', 'on_policy_pi_loss',
      'episodes', 'catch_rate_gain', 'probe_distance_change',
  ]
  starting = evaluation[evaluation.rule == 'starting_rule'][keys + metrics]
  learned = evaluation[evaluation.rule == 'learned_rule'][keys + metrics]
  paired = starting.merge(
      learned, on=keys, suffixes=('_starting', '_learned'), validate='one_to_one'
  )
  for name in ('catch_rate', 'mean_return', 'probe_policy_distance', 'on_policy_pi_loss'):
    paired[name + '_difference'] = paired[name + '_learned'] - paired[name + '_starting']
  summary = paired.groupby(['cohort', 'mode'], as_index=False).agg(
      n_cases=('experience_seed', 'size'),
      n_student_initialisations=('student_seed', 'nunique'),
      starting_catch_rate=('catch_rate_starting', 'mean'),
      learned_catch_rate=('catch_rate_learned', 'mean'),
      starting_catch_rate_gain=('catch_rate_gain_starting', 'mean'),
      learned_catch_rate_gain=('catch_rate_gain_learned', 'mean'),
      catch_rate_difference_mean=('catch_rate_difference', 'mean'),
      catch_rate_difference_std=('catch_rate_difference', 'std'),
      starting_probe_distance=('probe_policy_distance_starting', 'mean'),
      learned_probe_distance=('probe_policy_distance_learned', 'mean'),
      probe_distance_difference_mean=('probe_policy_distance_difference', 'mean'),
      probe_distance_difference_std=('probe_policy_distance_difference', 'std'),
  )
  configuration = pd.DataFrame([dict(
      inner_learning_rate=agent_settings.learning_rate,
      num_inner_steps=config.num_inner_steps, rollout_len=config.rollout_len,
      num_envs=config.num_envs, num_samples=trainer.agent.settings.update_rule.num_samples,
      use_auxiliary_losses=config.use_auxiliary_losses,
      eval_num_envs=config.eval_num_envs, eval_steps=config.eval_steps,
      original_student_seed=config.seeds[0], fresh_seeds=repr(fresh_seeds),
      experience_seeds=repr(experience_seeds), evaluation_seed=evaluation_seed,
      rule_parameter_difference_norm=_norm(jax.tree.map(
          lambda a, b: b - a, rules['starting_rule'], rules['learned_rule']
      )),
  )])
  fig, axes = plt.subplots(2, 3, figsize=(17, 8))
  order = ['untrained', 'starting_rule', 'learned_rule']
  labels = ['Untrained', 'Starting rule', 'Learned rule']
  for index, cohort in enumerate(('same_initialisation_new_data', 'fresh_initialisation_new_data')):
    for column, metric, mode in (
        (0, 'probe_policy_distance', 'student_stochastic'),
        (1, 'catch_rate', 'student_stochastic'),
        (2, 'catch_rate', 'student_mean_action'),
    ):
      frame = evaluation[(evaluation.cohort == cohort) & (evaluation['mode'] == mode)]
      for (student, experience), group in frame.groupby(['student_seed', 'experience_seed']):
        values = group.set_index('rule').loc[order, metric]
        axes[index, column].plot(
            labels, values, marker='o', label=f'student {student}, data {experience}'
        )
      title = 'Common-observation PPO distance' if column == 0 else (
          'Sampled-action Catch rate' if column == 1 else 'Mean-bank-action Catch rate'
      )
      axes[index, column].set(
          title=title, ylabel=cohort.replace('_', ' ') + '\n' + metric
      )
      if column > 0:
        axes[index, column].set_ylim(0, 1)
      axes[index, column].legend(fontsize=6)
  fig.tight_layout()
  return FrozenRuleTransferResult(evaluation, paired, summary, configuration, fig)


def save_frozen_rule_transfer(result: FrozenRuleTransferResult, output_dir: Path) -> None:
  output_dir.mkdir(parents=True, exist_ok=True)
  for name in ('evaluation', 'paired', 'summary', 'configuration'):
    getattr(result, name).to_csv(output_dir / f'{name}.csv', index=False)
  result.figure.savefig(output_dir / 'transfer_comparison.png', dpi=150)
  print(f'Frozen-rule transfer results saved to {output_dir}')
