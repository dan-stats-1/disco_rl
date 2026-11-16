"""Verify gradient geometry, measured steps and the real bank/notebook replay."""

import ast
from contextlib import redirect_stdout
from dataclasses import replace
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import jax
import jax.numpy as jnp
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from disco_rl.ppo import PPOConfig
from disco_rl.ppo import student_gradient_alignment as alignment
from disco_rl.ppo import train_ppo
from disco_rl.ppo.fixed_student_bank import fit_fixed_student_bank
from disco_rl.ppo.meta_training import MetaTrainingConfig
from disco_rl.ppo.meta_training import MetaTrainingTrainer
from disco_rl.ppo.test_imitation import small_agent_settings
from disco_rl.ppo.test_ppo import get_env


class GradientGeometryTest(unittest.TestCase):

  def test_exact_cosines_equal_steps_and_training_only_directions(self):
    targets = {
        11: jnp.array([-10.0, 0.0]), 15: jnp.array([1.0, -2.0]),
        12: jnp.array([0.0, 4.0]),
    }
    params = {'x': jnp.zeros(2)}
    original = np.array(params['x'], copy=True)
    differentiated = []

    def loss(p, seed):
      return 0.5 * (p['x']**2).sum() - (p['x'] * targets[seed]).sum()

    def gradient(p, seed):
      differentiated.append(seed)
      return jax.value_and_grad(lambda q: loss(q, seed))(p)

    students, pairs, directions, rows = alignment._measure_alignment(
        params, (11, 15), (12,), loss, gradient, (0.001, 0.01),
        checkpoint='toy', reference_losses={s: float(loss(params, s)) for s in targets},
        reference_tolerance=1e-6, loss_tolerance=1e-7,
    )
    self.assertEqual(differentiated, [11, 15])
    np.testing.assert_array_equal(params['x'], original)
    np.testing.assert_allclose(pairs[0]['cosine'], -1 / np.sqrt(5), rtol=1e-6)
    self.assertTrue(all(d['tested'] for d in directions))
    self.assertTrue(np.isnan(students[-1]['gradient_norm']))
    trials = pd.DataFrame(rows)
    self.assertEqual(len(trials), 24)
    np.testing.assert_allclose(trials.actual_step_norm, trials.requested_step_norm, rtol=1e-6)
    self.assertTrue(trials[trials.split == 'validation'].predicted_change.isna().all())
    # Both directions use the actual mean-loss gradient for predictions, not the
    # norm-balanced vector as a fictitious derivative of the unweighted loss.
    train = trials[trials.split == 'training']
    np.testing.assert_allclose(
        train.loss_change - train.predicted_change,
        0.5 * train.actual_step_norm**2, atol=2e-8,
    )
    smaller_student = trials[
        (trials.seed == 15) & (trials.sign == -1)
        & (trials.requested_step_norm == 0.01)
    ].set_index('direction')
    self.assertGreater(smaller_student.loc['raw_mean', 'loss_change'], 0)
    self.assertLess(smaller_student.loc['norm_balanced_mean', 'loss_change'], 0)
    for direction, group in trials.groupby('direction'):
      for step, step_group in group.groupby('requested_step_norm'):
        paired = step_group[step_group.split == 'training'].groupby('sign').loss_change.mean()
        self.assertLess(paired.loc[-1], paired.loc[1], msg=f'{direction}, {step}')

  def test_zero_and_cancelling_gradients_are_reported(self):
    params = {'x': jnp.zeros(1)}

    def evaluate(p, _seed):
      return (p['x']**2).sum()

    for gradients in ((0.0, 0.0), (1.0, -1.0)):
      with self.subTest(gradients=gradients):
        _, pairs, directions, trials = alignment._measure_alignment(
            params, (11, 15), (12,), evaluate,
            lambda p, s: (evaluate(p, s), {'x': jnp.array([gradients[0 if s == 11 else 1]])}),
            (0.01,), checkpoint='zero', reference_losses={11: 0.0, 15: 0.0, 12: 0.0},
            reference_tolerance=1e-6, loss_tolerance=1e-7,
        )
        self.assertEqual(trials, [])
        self.assertTrue(all(d['status'] == 'unavailable_zero_norm' for d in directions))
        if gradients[0] == 0:
          self.assertTrue(np.isnan(pairs[0]['cosine']))
        else:
          self.assertEqual(pairs[0]['cosine'], -1)

  def test_invalid_steps_reference_mismatch_and_nonfinite_gradients(self):
    params = {'x': jnp.zeros(1)}
    common = dict(
        checkpoint='guard', reference_losses={11: 0.0, 15: 0.0, 12: 0.0},
        reference_tolerance=1e-6, loss_tolerance=1e-7,
    )
    for steps in ((), (-0.1,), (np.nan,), (0.1, 0.1)):
      with self.subTest(steps=steps), self.assertRaises(ValueError):
        alignment._measure_alignment(
            params, (11, 15), (12,), lambda _p, _s: jnp.array(0.0),
            lambda p, _s: (jnp.array(0.0), p), steps, **common,
        )
    with self.assertRaisesRegex(RuntimeError, 'does not match the saved bank loss'):
      alignment._measure_alignment(
          params, (11, 15), (12,), lambda _p, _s: jnp.array(0.1),
          lambda p, _s: (jnp.array(0.1), p), (0.01,), **common,
      )
    with self.assertRaises(FloatingPointError):
      alignment._measure_alignment(
          params, (11, 15), (12,), lambda _p, _s: jnp.array(0.0),
          lambda _p, _s: (jnp.array(0.0), {'x': jnp.array([np.nan])}), (0.01,), **common,
      )


class RealStudentAlignmentTest(unittest.TestCase):

  def tearDown(self):
    plt.close('all')

  def test_bank_replay_is_immutable_and_notebook_exports_are_wired(self):
    with redirect_stdout(StringIO()):
      teacher = train_ppo(get_env, PPOConfig(
          seeds=(302,), num_updates=1, num_envs=1, horizon=8,
          epochs=1, num_minibatches=1, eval_every=1, eval_num_envs=1,
          eval_steps=7, checkpoint_steps=(),
      ))
    settings = small_agent_settings()
    settings.learning_rate = 0.0005
    config = MetaTrainingConfig(
        seeds=(811,), num_updates=1, num_inner_steps=1, num_envs=1,
        rollout_len=8, eval_num_envs=1, eval_steps=7,
        fresh_seeds=(812,), fresh_updates=1,
    )
    trainer = MetaTrainingTrainer(teacher, settings, config)
    eta, _ = trainer.agent.update_rule.init_params(jax.random.PRNGKey(501))
    with redirect_stdout(StringIO()):
      bank = fit_fixed_student_bank(
          teacher, settings, config, meta_params=eta,
          training_seeds=(811, 815), validation_seeds=(812,),
          reserved_test_seeds=(831,), step_multipliers=(1.0,),
      )
    state = (
        bank.optimisation.initial_meta_params, bank.optimisation.final_meta_params,
        bank.optimisation.final_meta_opt_state, teacher.final_params,
    )
    before = jax.tree.map(lambda x: np.array(x, copy=True), state)
    collected = []
    snapshots = []
    fixed = alignment._fixed_snapshot
    collect = alignment._collect_fixed_snapshot

    def capture(seed, snapshot):
      collected.append(seed)
      snapshots.append((snapshot, jax.tree.map(lambda x: np.array(x, copy=True), snapshot)))
      return snapshot

    def first(*args):
      result = fixed(*args)
      capture(811, result[2:])
      return result

    def remaining(trainer, seed):
      return capture(seed, collect(trainer, seed))

    with (
        patch.object(alignment, '_fixed_snapshot', side_effect=first),
        patch.object(alignment, '_collect_fixed_snapshot', side_effect=remaining),
        redirect_stdout(StringIO()),
    ):
      result = alignment.probe_student_gradient_alignment(
          teacher, settings, config, bank, step_norms=(0.001,),
      )
    self.assertEqual(collected, [811, 815, 812])
    self.assertEqual(len(result.students), 6)
    self.assertEqual(len(result.pairs), 2)
    self.assertEqual(set(result.students.checkpoint), {'initial', 'final'})
    self.assertTrue(result.students[result.students.split == 'validation'].gradient_norm.isna().all())
    self.assertEqual(len(result.trials), 24)
    self.assertEqual(len(result.summary), 16)
    for old, current in zip(jax.tree.leaves(before), jax.tree.leaves(state)):
      np.testing.assert_array_equal(old, current)
    for snapshot, original in snapshots:
      for old, current in zip(jax.tree.leaves(original), jax.tree.leaves(snapshot)):
        np.testing.assert_array_equal(old, current)
    with patch.object(alignment, '_fixed_snapshot') as collect_mock:
      with self.assertRaisesRegex(RuntimeError, 'rollout_len changed'):
        alignment.probe_student_gradient_alignment(
            teacher, settings, replace(config, rollout_len=16), bank,
        )
      collect_mock.assert_not_called()
    notebook = json.loads((
        Path(__file__).resolve().parents[2] / 'colabs' / 'meta_train.ipynb'
    ).read_text(encoding='utf-8'))
    index = next(
        i for i, c in enumerate(notebook['cells'])
        if ''.join(c['source']).startswith(
            '# @title Diagnose student-gradient conflict and norm imbalance.'
        )
    )
    self.assertEqual(notebook['cells'][index - 1]['cell_type'], 'markdown')
    source = ''.join(notebook['cells'][index]['source'])
    with tempfile.TemporaryDirectory() as directory:
      namespace = dict(
          ppo_result=teacher, bank_agent_settings=settings, bank_config=config,
          bank_result=bank, repo_root=directory,
      )
      with (
          patch('importlib.reload', side_effect=lambda module: module),
          patch.object(
              alignment, 'probe_student_gradient_alignment', return_value=result,
          ) as run,
          patch('IPython.display.display'), patch.object(plt, 'show'),
      ):
        exec(compile(ast.parse(source), '<gradient-alignment-notebook-cell>', 'exec'), namespace)
      self.assertIs(run.call_args.args[3], bank)
      self.assertEqual(run.call_args.kwargs['step_norms'], (1e-4, 1e-3, 1e-2))
      output = Path(directory) / 'results' / 'continuous_catch_student_gradient_alignment'
      for name in (
          'students.csv', 'gradient_pairs.csv', 'direction_status.csv',
          'trials.csv', 'summary.csv', 'configuration.csv', 'gradient_alignment.png',
      ):
        self.assertGreater((output / name).stat().st_size, 0)


if __name__ == '__main__':
  unittest.main()
