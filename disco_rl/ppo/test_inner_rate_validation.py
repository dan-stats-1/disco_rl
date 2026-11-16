"""Verify shared-data inner-rate comparisons and their notebook wrapper."""

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
from matplotlib import pyplot as plt
import numpy as np

from disco_rl.ppo import PPOConfig
from disco_rl.ppo import train_ppo
from disco_rl.ppo import inner_rate_validation
from disco_rl.ppo.inner_rate_validation import probe_inner_learning_rates
from disco_rl.ppo.meta_training import MetaTrainingConfig
from disco_rl.ppo.meta_training import MetaTrainingTrainer
from disco_rl.ppo.meta_update_validation import _fixed_snapshot
from disco_rl.ppo.test_imitation import small_agent_settings
from disco_rl.ppo.test_ppo import get_env
from disco_rl.ppo.training import PPOResult


class InnerRateValidationTest(unittest.TestCase):
  teacher: PPOResult

  @classmethod
  def setUpClass(cls):
    with redirect_stdout(StringIO()):
      cls.teacher = train_ppo(get_env, PPOConfig(
          seeds=(302,), num_updates=1, num_envs=1, horizon=8,
          epochs=1, num_minibatches=1, eval_every=1, eval_num_envs=1,
          eval_steps=7, checkpoint_steps=(),
      ))

  def setUp(self):
    self.settings = small_agent_settings()
    self.settings.learning_rate = 0.05
    self.config = MetaTrainingConfig(
        seeds=(811,), num_updates=100, num_inner_steps=1, num_envs=1,
        rollout_len=8, eval_num_envs=1, eval_steps=7,
        fresh_seeds=(812,), fresh_updates=1,
    )

  def tearDown(self):
    plt.close('all')

  def test_invalid_inputs_fail_before_collecting(self):
    with patch.object(inner_rate_validation, '_fixed_snapshot') as collect:
      for rates in ((), (0,), (-0.1,), (np.nan,), (0.05, 0.05)):
        with self.subTest(rates=rates), self.assertRaises(ValueError):
          probe_inner_learning_rates(
              self.teacher, self.settings, self.config, learning_rates=rates
          )
      with self.assertRaises(ValueError):
        probe_inner_learning_rates(
            self.teacher, self.settings, self.config, step_norms=(0,)
        )
      collect.assert_not_called()

  def test_shared_snapshot_is_immutable_and_notebook_exports(self):
    snapshots, replays = [], []
    original_outer = MetaTrainingTrainer.outer_loss
    teacher_before = jax.tree.map(lambda x: np.array(x, copy=True), self.teacher.final_params)

    def capture(*args):
      snapshot = _fixed_snapshot(*args)
      snapshots.append((snapshot[1:], jax.tree.map(
          lambda x: np.array(x, copy=True), snapshot[1:]
      )))
      return snapshot

    def record(trainer, eta, learner, train, fitting, key):
      replays.append((
          trainer.agent.settings.learning_rate,
          id(learner), id(train), id(fitting), id(key),
      ))
      return original_outer(trainer, eta, learner, train, fitting, key)

    with patch.object(inner_rate_validation, '_fixed_snapshot', side_effect=capture) as collect:
      with patch.object(MetaTrainingTrainer, 'outer_loss', record), redirect_stdout(StringIO()):
        result = probe_inner_learning_rates(
            self.teacher, self.settings, self.config,
            learning_rates=(0.005,), step_norms=(1e-3,),
        )
      collect.assert_called_once()
    self.assertEqual(list(result.summary.inner_learning_rate), [0.05, 0.005])
    self.assertEqual(len(result.trials), 10)
    self.assertEqual(len(result.slopes), 4)
    self.assertEqual(set(r[0] for r in replays), {0.05, 0.005})
    self.assertEqual(len(set(r[1:] for r in replays)), 1)
    self.assertTrue((result.summary.outer_learning_rate == self.config.meta_learning_rate).all())
    self.assertTrue((result.summary.num_inner_steps == 1).all())
    self.assertTrue((result.summary.replay_error == 0).all())
    self.assertEqual(result.summary.policy_loss_before_inner.nunique(), 1)
    self.assertTrue(np.isfinite(result.summary.policy_loss_after_inner).all())
    snapshot, before = snapshots[0]
    for old, current in zip(jax.tree.leaves(before), jax.tree.leaves(snapshot)):
      np.testing.assert_array_equal(old, current)
    for old, current in zip(jax.tree.leaves(teacher_before), jax.tree.leaves(self.teacher.final_params)):
      np.testing.assert_array_equal(old, current)
    self.assertEqual(self.settings.learning_rate, 0.05)
    self.assertEqual(self.settings.hyper_params.y_cost, 1)

    notebook_path = Path(__file__).resolve().parents[2] / 'colabs' / 'meta_train.ipynb'
    notebook = json.loads(notebook_path.read_text(encoding='utf-8'))
    source = next(
        ''.join(cell['source']) for cell in notebook['cells']
        if ''.join(cell['source']).startswith(
            '# @title Compare inner learning rates without training the meta-network again.'
        )
    )
    self.assertIn('probe_inner_learning_rates', source)
    baseline = float(result.summary.policy_loss_after_inner.iloc[0])
    with tempfile.TemporaryDirectory() as directory:
      namespace = dict(
          ppo_result=self.teacher, agent_settings=self.settings, fixed_meta_config=self.config,
          fixed_meta_result=SimpleNamespace(
              final_meta_params=snapshot[0],
              training=result.summary.rename(columns={'policy_loss_after_inner': 'fit_loss'}).iloc[:1],
          ),
          repo_root=directory,
      )
      with (
          patch('importlib.reload', side_effect=lambda module: module),
          patch.object(inner_rate_validation, 'probe_inner_learning_rates', return_value=result) as probe,
          patch('IPython.display.display'), patch.object(plt, 'show'),
          redirect_stdout(StringIO()),
      ):
        exec(compile(ast.parse(source), '<inner-rate-notebook-cell>', 'exec'), namespace)
      self.assertEqual(probe.call_args.kwargs['expected_baseline_loss'], baseline)
      self.assertIs(probe.call_args.kwargs['meta_params'], snapshot[0])
      output = Path(directory) / 'results' / 'continuous_catch_inner_rate_probe'
      for name in ('summary.csv', 'trials.csv', 'slopes.csv', 'direction_status.csv', 'learning_rate_comparison.png'):
        self.assertGreater((output / name).stat().st_size, 0)

    with redirect_stdout(StringIO()):
      with self.assertRaisesRegex(RuntimeError, 'Original-rate checkpoint replay does not match'):
        probe_inner_learning_rates(
            self.teacher, self.settings, self.config, learning_rates=(0.005,),
            step_norms=(1e-3,), expected_baseline_loss=baseline + 0.1,
        )


if __name__ == '__main__':
  unittest.main()
