"""Verify paired frozen-rule controls, immutable snapshots and notebook exports."""

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
import pandas as pd

from disco_rl.ppo import PPOConfig
from disco_rl.ppo import train_ppo
from disco_rl.ppo import frozen_rule_transfer
from disco_rl.ppo.frozen_rule_transfer import compare_frozen_rules
from disco_rl.ppo.meta_training import MetaTrainingConfig
from disco_rl.ppo.meta_training import MetaTrainingTrainer
from disco_rl.ppo.test_imitation import small_agent_settings
from disco_rl.ppo.test_ppo import get_env
from disco_rl.ppo.training import PPOResult


class FrozenRuleTransferTest(unittest.TestCase):
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
    self.settings.learning_rate = 0.0005
    self.config = MetaTrainingConfig(
        seeds=(811,), num_updates=100, num_inner_steps=2, num_envs=1,
        rollout_len=8, eval_num_envs=1, eval_steps=14,
        fresh_seeds=(812,), fresh_updates=1,
    )
    trainer = MetaTrainingTrainer(self.teacher, self.settings, self.config)
    self.eta, _ = trainer.agent.update_rule.init_params(jax.random.PRNGKey(501))

  def tearDown(self):
    plt.close('all')

  def test_invalid_seeds_and_rule_architecture(self):
    for kwargs in (
        dict(fresh_seeds=(811,), experience_seeds=(921,)),
        dict(fresh_seeds=(812,), experience_seeds=(811,)),
        dict(fresh_seeds=(812, 813), experience_seeds=(921,)),
        dict(fresh_seeds=(812, 812), experience_seeds=(921, 922)),
        dict(fresh_seeds=(812,), experience_seeds=(-1,)),
    ):
      with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
        compare_frozen_rules(
            self.teacher, self.settings, self.config, self.eta, self.eta, **kwargs
        )
    with self.assertRaisesRegex(ValueError, 'learned_rule does not match'):
      compare_frozen_rules(
          self.teacher, self.settings, self.config, self.eta, {},
          fresh_seeds=(812,), experience_seeds=(921,),
      )

  def test_identical_rules_are_paired_and_inputs_are_immutable(self):
    frozen = jax.tree.map(
        lambda x: np.array(x, copy=True), (self.eta, self.teacher.final_params)
    )
    snapshots = []
    original = frozen_rule_transfer._collect_case

    def collect(trainer, learner, seed):
      train, probe, key = original(trainer, learner, seed)
      snapshot = learner, train, probe, key
      snapshots.append((seed, snapshot, jax.tree.map(
          lambda x: np.array(x, copy=True), snapshot
      )))
      return train, probe, key

    with (
        patch.object(frozen_rule_transfer, '_collect_case', side_effect=collect) as collector,
        redirect_stdout(StringIO()),
    ):
      result = compare_frozen_rules(
          self.teacher, self.settings, self.config, self.eta, self.eta,
          fresh_seeds=(812,), experience_seeds=(921,),
      )
    self.assertEqual(collector.call_count, 2)
    self.assertEqual(len(result.evaluation), 12)
    self.assertEqual(len(result.paired), 4)
    self.assertEqual(len(result.summary), 4)
    self.assertEqual(set(result.evaluation.student_seed), {811, 812})
    self.assertEqual(set(result.evaluation.experience_seed), {921})
    self.assertEqual(set(result.evaluation.agent_updates), {0, 2})
    after = result.evaluation[result.evaluation.stage == 'after']
    self.assertTrue((after.training_env_steps == 16).all())
    self.assertTrue((after.episodes > 0).all())
    for metric in ('catch_rate', 'mean_return', 'probe_policy_distance', 'on_policy_pi_loss'):
      np.testing.assert_array_equal(result.paired[metric + '_difference'], np.zeros(4))
    for _, snapshot, before in snapshots:
      for old, current in zip(jax.tree.leaves(before), jax.tree.leaves(snapshot)):
        np.testing.assert_array_equal(old, current)
    for old, current in zip(
        jax.tree.leaves(frozen), jax.tree.leaves((self.eta, self.teacher.final_params))
    ):
      np.testing.assert_array_equal(old, current)
    self.assertTrue(any(
        not np.array_equal(a, b)
        for a, b in zip(
            jax.tree.leaves(snapshots[0][1][0].params),
            jax.tree.leaves(snapshots[1][1][0].params),
        )
    ))
    self.assertEqual(self.settings.learning_rate, 0.0005)
    self.assertEqual(self.settings.hyper_params.y_cost, 1)

    notebook = json.loads((
        Path(__file__).resolve().parents[2] / 'colabs' / 'meta_train.ipynb'
    ).read_text(encoding='utf-8'))
    source = next(
        ''.join(c['source']) for c in notebook['cells']
        if ''.join(c['source']).startswith(
            '# @title Compare frozen starting and learned rules on new data and fresh students.'
        )
    )
    metadata = pd.DataFrame([dict(
        inner_learning_rate=0.0005, outer_learning_rate=self.config.meta_learning_rate,
        num_inner_steps=self.config.num_inner_steps, rollout_len=self.config.rollout_len,
        num_envs=self.config.num_envs, num_samples=self.settings.update_rule.num_samples,
    )])
    with tempfile.TemporaryDirectory() as directory:
      namespace = dict(
          ppo_result=self.teacher, low_rate_agent_settings=self.settings,
          low_rate_meta_config=self.config, low_rate_meta_summary=metadata,
          low_rate_meta_result=SimpleNamespace(
              initial_meta_params=self.eta, final_meta_params=self.eta,
          ), repo_root=directory,
      )
      with (
          patch('importlib.reload', side_effect=lambda module: module),
          patch.object(
              frozen_rule_transfer, 'compare_frozen_rules', return_value=result,
          ) as compare,
          patch('IPython.display.display'), patch.object(plt, 'show'),
          redirect_stdout(StringIO()),
      ):
        exec(compile(ast.parse(source), '<frozen-transfer-notebook-cell>', 'exec'), namespace)
      self.assertEqual(compare.call_args.args[2].num_inner_steps, 2)
      self.assertEqual(compare.call_args.args[2].eval_num_envs, 16)
      self.assertEqual(compare.call_args.args[2].eval_steps, 140)
      self.assertIs(compare.call_args.args[3], self.eta)
      self.assertIs(compare.call_args.args[4], self.eta)
      self.assertEqual(self.config.eval_num_envs, 1)
      output = Path(directory) / 'results' / 'continuous_catch_frozen_rule_transfer'
      for name in ('evaluation.csv', 'paired.csv', 'summary.csv', 'configuration.csv', 'transfer_comparison.png'):
        self.assertGreater((output / name).stat().st_size, 0)


if __name__ == '__main__':
  unittest.main()
