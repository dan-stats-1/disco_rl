"""Verify exact training-gradient averaging, split isolation and bank replay."""

import ast
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import jax
import jax.numpy as jnp
from matplotlib import pyplot as plt
import numpy as np
import optax
import pandas as pd

from disco_rl.ppo import PPOConfig
from disco_rl.ppo import train_ppo
from disco_rl.ppo import fixed_student_bank
from disco_rl.ppo import frozen_rule_transfer
from disco_rl.ppo.fixed_meta_training import _optimise_fixed_loss
from disco_rl.ppo.fixed_student_bank import _sequential_objectives
from disco_rl.ppo.fixed_student_bank import fit_fixed_student_bank
from disco_rl.ppo.frozen_rule_transfer import FrozenRuleTransferResult
from disco_rl.ppo.meta_training import MetaTrainingConfig
from disco_rl.ppo.meta_training import MetaTrainingTrainer
from disco_rl.ppo.meta_update_validation import _collect_fixed_snapshot
from disco_rl.ppo.meta_update_validation import _fixed_snapshot
from disco_rl.ppo.test_imitation import small_agent_settings
from disco_rl.ppo.test_ppo import get_env


class SequentialBankGradientTest(unittest.TestCase):

  def tearDown(self):
    plt.close('all')

  def test_exact_mean_gradient_and_no_validation_gradients(self):
    targets = {11: 1.0, 15: 3.0, 12: -10.0}
    differentiated = []
    evaluated = []
    scalar = jax.jit(lambda p, target: ((p['x'] - target)**2).sum())
    derivative = jax.jit(jax.value_and_grad(scalar))

    def evaluate(params, seed):
      evaluated.append(seed)
      return scalar(params, targets[seed])

    def gradient(params, seed):
      differentiated.append(seed)
      return derivative(params, targets[seed])

    losses, gradients, rows, record_final = _sequential_objectives(
        (11, 15), (12,), evaluate, gradient
    )
    initial = {'x': jnp.zeros(1)}
    with redirect_stdout(StringIO()):
      result = _optimise_fixed_loss(
          initial, losses, optax.sgd(0.1), 2, (0.1, 1), 1, 1e-7, 1,
          gradient_fn=gradients,
      )
    record_final(result.final_meta_params)
    np.testing.assert_allclose(result.final_meta_params['x'], [0.72], rtol=1e-6)
    np.testing.assert_allclose(result.training.gradient_norm_before.iloc[1:], [4, 3.2], rtol=1e-6)
    self.assertEqual(differentiated, [11, 15, 11, 15])
    self.assertEqual(set(evaluated), {11, 15, 12})
    self.assertGreater(result.training.heldout_loss.iloc[-1], result.training.heldout_loss.iloc[0])
    self.assertEqual(result.replay_error, 0)
    table = pd.DataFrame(rows)
    self.assertEqual(len(table), 9)
    self.assertTrue(table[table.split == 'validation'].gradient_norm_for_next_update.isna().all())
    fits = table[table.split == 'training'].groupby('update').policy_distance.mean()
    np.testing.assert_allclose(fits, result.training.fit_loss, rtol=1e-6)
    np.testing.assert_array_equal(initial['x'], [0])


class FixedStudentBankTest(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    with redirect_stdout(StringIO()):
      cls.teacher = train_ppo(get_env, PPOConfig(
          seeds=(302,), num_updates=1, num_envs=1, horizon=8,
          epochs=1, num_minibatches=1, eval_every=1, eval_num_envs=1,
          eval_steps=7, checkpoint_steps=(),
      ))
    cls.settings = small_agent_settings()
    cls.settings.learning_rate = 0.0005
    cls.config = MetaTrainingConfig(
        seeds=(811,), num_updates=2, num_inner_steps=1, num_envs=1,
        rollout_len=8, eval_num_envs=1, eval_steps=7,
        fresh_seeds=(812,), fresh_updates=1,
    )
    trainer = MetaTrainingTrainer(cls.teacher, cls.settings, cls.config)
    cls.eta, _ = trainer.agent.update_rule.init_params(jax.random.PRNGKey(501))

  def tearDown(self):
    plt.close('all')

  def test_split_overlap_fails_before_collection(self):
    with patch.object(fixed_student_bank, '_fixed_snapshot') as collect:
      for kwargs in (
          dict(training_seeds=(811,), validation_seeds=(812,), reserved_test_seeds=(831,)),
          dict(training_seeds=(811, 815), validation_seeds=(815,), reserved_test_seeds=(831,)),
          dict(training_seeds=(811, 815), validation_seeds=(812,), reserved_test_seeds=(812,)),
          dict(training_seeds=(815, 816), validation_seeds=(812,), reserved_test_seeds=(831,)),
      ):
        with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
          fit_fixed_student_bank(
              self.teacher, self.settings, self.config, meta_params=self.eta, **kwargs
          )
      collect.assert_not_called()

  def test_real_bank_is_reset_and_reserved_students_untouched(self):
    snapshots = []
    frozen = jax.tree.map(
        lambda x: np.array(x, copy=True), (self.eta, self.teacher.final_params)
    )

    def capture(seed, snapshot):
      snapshots.append((seed, snapshot, jax.tree.map(
          lambda x: np.array(x, copy=True), snapshot
      )))
      return snapshot

    def initial(*args):
      snapshot = _fixed_snapshot(*args)
      capture(811, snapshot[2:])
      return snapshot

    def collect(trainer, seed):
      return capture(seed, _collect_fixed_snapshot(trainer, seed))

    with (
        patch.object(fixed_student_bank, '_fixed_snapshot', side_effect=initial),
        patch.object(fixed_student_bank, '_collect_fixed_snapshot', side_effect=collect),
        redirect_stdout(StringIO()),
    ):
      result = fit_fixed_student_bank(
          self.teacher, self.settings, self.config, meta_params=self.eta,
          training_seeds=(811, 815), validation_seeds=(812,),
          reserved_test_seeds=(831,), step_multipliers=(0.1, 1),
      )
    self.assertEqual([s[0] for s in snapshots], [811, 815, 812])
    self.assertEqual(len(result.student_history), 9)
    self.assertEqual(len(result.optimisation.training), 3)
    self.assertEqual(int(result.optimisation.final_meta_opt_state[1][0].count), 2)
    self.assertEqual(result.optimisation.replay_error, 0)
    self.assertEqual(set(result.student_history.seed), {811, 815, 812})
    for seed, snapshot, before in snapshots:
      for old, current in zip(jax.tree.leaves(before), jax.tree.leaves(snapshot)):
        np.testing.assert_array_equal(old, current, err_msg=f'student {seed}')
    for old, current in zip(
        jax.tree.leaves(frozen), jax.tree.leaves((self.eta, self.teacher.final_params))
    ):
      np.testing.assert_array_equal(old, current)
    self.assertEqual(self.settings.learning_rate, 0.0005)
    self.assertEqual(self.settings.hyper_params.y_cost, 1)

    notebook = json.loads((
        Path(__file__).resolve().parents[2] / 'colabs' / 'meta_train.ipynb'
    ).read_text(encoding='utf-8'))
    source = next(
        ''.join(c['source']) for c in notebook['cells']
        if ''.join(c['source']).startswith(
            '# @title Fit a rule over a fixed bank of reset student initialisations.'
        )
    )
    original_loss = result.student_history[
        (result.student_history.seed == 811) & (result.student_history['update'] == 0)
    ].policy_distance.iloc[0]
    with redirect_stdout(StringIO()):
      with self.assertRaisesRegex(RuntimeError, 'Original-student pre-fit loss does not match'):
        fit_fixed_student_bank(
            self.teacher, self.settings, self.config, meta_params=self.eta,
            training_seeds=(811, 815), validation_seeds=(812,),
            reserved_test_seeds=(831,), expected_original_fit_loss=original_loss + 0.1,
        )
    metadata = pd.DataFrame([dict(
        inner_learning_rate=0.0005, outer_learning_rate=self.config.meta_learning_rate,
        num_inner_steps=self.config.num_inner_steps, rollout_len=self.config.rollout_len,
        num_envs=self.config.num_envs, num_samples=self.settings.update_rule.num_samples,
        initial_fit_loss=original_loss,
    )])
    frame = pd.DataFrame([dict(value=1)])
    transfer = FrozenRuleTransferResult(frame, frame, frame, frame, plt.figure())
    with tempfile.TemporaryDirectory() as directory:
      namespace = dict(
          ppo_result=self.teacher, low_rate_agent_settings=self.settings,
          low_rate_meta_config=self.config, low_rate_meta_summary=metadata,
          low_rate_meta_result=SimpleNamespace(
              initial_meta_params=self.eta,
              final_meta_params=jax.tree.map(lambda x: x + 0.1, self.eta),
          ), repo_root=directory,
      )
      with (
          patch('importlib.reload', side_effect=lambda module: module),
          patch.object(fixed_student_bank, 'fit_fixed_student_bank', return_value=result) as fit,
          patch.object(frozen_rule_transfer, 'compare_frozen_rules', return_value=transfer) as compare,
          patch('IPython.display.display'), patch.object(plt, 'show'),
          redirect_stdout(StringIO()),
      ):
        exec(compile(ast.parse(source), '<fixed-bank-notebook-cell>', 'exec'), namespace)
      self.assertIs(fit.call_args.kwargs['meta_params'], self.eta)
      self.assertEqual(fit.call_args.kwargs['expected_original_fit_loss'], original_loss)
      self.assertEqual(fit.call_args.kwargs['training_seeds'], (811, 815, 816, 817))
      self.assertEqual(fit.call_args.kwargs['validation_seeds'], (812, 813, 814))
      self.assertEqual(fit.call_args.kwargs['reserved_test_seeds'], (831, 832, 833))
      self.assertEqual(fit.call_args.args[2].num_updates, 100)
      self.assertIs(compare.call_args.args[3], result.optimisation.initial_meta_params)
      self.assertIs(compare.call_args.args[4], result.optimisation.final_meta_params)
      self.assertEqual(compare.call_args.kwargs['fresh_seeds'], (812, 813, 814))
      self.assertEqual(self.config.num_updates, 2)
      output = Path(directory) / 'results' / 'continuous_catch_fixed_student_bank'
      for name in (
          'training.csv', 'step_trials.csv', 'summary.csv', 'learning_curves.png',
          'students.csv', 'configuration.csv', 'student_curves.png',
          'validation_transfer\\paired.csv',
      ):
        self.assertGreater((output / name).stat().st_size, 0)


if __name__ == '__main__':
  unittest.main()
