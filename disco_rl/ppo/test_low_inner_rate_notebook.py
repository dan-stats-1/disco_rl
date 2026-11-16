"""Exercise the low-inner-rate notebook experiment with a tiny real learner."""

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
from ml_collections import config_dict
import numpy as np
import pandas as pd

from disco_rl.ppo import PPOConfig
from disco_rl.ppo import train_ppo
from disco_rl.ppo import fixed_meta_training
from disco_rl.ppo.meta_training import MetaTrainingConfig
from disco_rl.ppo.meta_update_validation import _fixed_snapshot
from disco_rl.ppo.test_imitation import small_agent_settings
from disco_rl.ppo.test_ppo import get_env


class LowInnerRateNotebookTest(unittest.TestCase):

  def tearDown(self):
    plt.close('all')

  def test_real_notebook_run_preserves_inputs_and_matches_sweep(self):
    with redirect_stdout(StringIO()):
      teacher = train_ppo(get_env, PPOConfig(
          seeds=(302,), num_updates=1, num_envs=1, horizon=8,
          epochs=1, num_minibatches=1, eval_every=1, eval_num_envs=1,
          eval_steps=7, checkpoint_steps=(),
      ))
    settings = small_agent_settings()
    settings.learning_rate = 0.05
    config = MetaTrainingConfig(
        seeds=(811,), num_updates=7, num_inner_steps=1, num_envs=1,
        rollout_len=8, eval_num_envs=1, eval_steps=7,
        fresh_seeds=(812,), fresh_updates=1,
    )
    smaller = config_dict.ConfigDict(settings.to_dict())
    smaller.learning_rate = 0.0005
    trainer, eta, learner, train, fitting, key = _fixed_snapshot(
        teacher, smaller, config, None
    )
    expected = float(jax.jit(trainer.outer_loss)(eta, learner, train, fitting, key)[0])
    frozen = jax.tree.map(lambda x: np.array(x, copy=True), (eta, teacher.final_params))
    sweep = pd.DataFrame([dict(
        inner_learning_rate=0.0005, policy_loss_after_inner=expected,
        outer_learning_rate=config.meta_learning_rate,
        num_inner_steps=config.num_inner_steps, rollout_len=config.rollout_len,
        num_envs=config.num_envs, num_samples=settings.update_rule.num_samples,
    )])
    path = Path(__file__).resolve().parents[2] / 'colabs' / 'meta_train.ipynb'
    notebook = json.loads(path.read_text(encoding='utf-8'))
    source = next(
        ''.join(cell['source']) for cell in notebook['cells']
        if ''.join(cell['source']).startswith(
            '# @title Repeat fixed-problem meta-training with the validated smaller inner rate.'
        )
    )
    self.assertIn('low_rate_meta_updates = 100', source)

    class ReduceBudget(ast.NodeTransformer):
      def visit_Assign(self, node):
        if any(isinstance(t, ast.Name) and t.id == 'low_rate_meta_updates' for t in node.targets):
          node.value = ast.Constant(2)
        return node

    code = compile(
        ast.fix_missing_locations(ReduceBudget().visit(ast.parse(source))),
        '<low-inner-rate-notebook-cell>', 'exec',
    )
    with tempfile.TemporaryDirectory() as directory:
      namespace = dict(
          ppo_result=teacher, agent_settings=settings, fixed_meta_config=config,
          fixed_meta_result=SimpleNamespace(final_meta_params=eta),
          inner_rate_result=SimpleNamespace(summary=sweep), repo_root=directory,
      )
      with (
          patch('importlib.reload', side_effect=lambda module: module),
          patch.object(
              fixed_meta_training, 'fit_fixed_meta',
              wraps=fixed_meta_training.fit_fixed_meta,
          ) as fit,
          patch('IPython.display.display'), patch.object(plt, 'show'),
          redirect_stdout(StringIO()),
      ):
        exec(code, namespace)
      fit.assert_called_once()
      self.assertEqual(fit.call_args.kwargs['expected_initial_fit_loss'], expected)
      self.assertIs(fit.call_args.kwargs['meta_params'], eta)
      result = namespace['low_rate_meta_result']
      self.assertEqual(len(result.training), 3)
      self.assertEqual(len(result.step_trials), 8)
      self.assertAlmostEqual(result.training.fit_loss.iloc[0], expected, delta=1e-6)
      self.assertEqual(int(result.final_meta_opt_state[1][0].count), 2)
      self.assertTrue(np.isfinite(result.training.select_dtypes('number')).all().all())
      self.assertEqual(namespace['low_rate_agent_settings'].learning_rate, 0.0005)
      self.assertEqual(namespace['low_rate_meta_config'].meta_learning_rate, config.meta_learning_rate)
      self.assertEqual(settings.learning_rate, 0.05)
      self.assertEqual(config.num_updates, 7)
      for old, current in zip(jax.tree.leaves(frozen), jax.tree.leaves((eta, teacher.final_params))):
        np.testing.assert_array_equal(old, current)
      output = Path(directory) / 'results' / 'continuous_catch_fixed_meta_low_inner_rate'
      for name in (
          'training.csv', 'step_trials.csv', 'summary.csv', 'learning_curves.png',
          'configuration_and_outcome.csv',
      ):
        self.assertGreater((output / name).stat().st_size, 0)
      saved = pd.read_csv(output / 'configuration_and_outcome.csv')
      self.assertEqual(saved.inner_learning_rate.iloc[0], 0.0005)
      self.assertAlmostEqual(saved.sweep_fit_loss.iloc[0], saved.initial_fit_loss.iloc[0], delta=1e-6)

      sweep.loc[0, 'num_samples'] = 999
      with (
          patch('importlib.reload', side_effect=lambda module: module),
          patch.object(fixed_meta_training, 'fit_fixed_meta') as blocked,
          self.assertRaisesRegex(RuntimeError, 'num_samples changed'),
      ):
        exec(code, namespace)
      blocked.assert_not_called()


if __name__ == '__main__':
  unittest.main()
