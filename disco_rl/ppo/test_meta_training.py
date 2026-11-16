"""Regression tests for the nested PPO-guided meta-gradient pathway."""

from contextlib import redirect_stdout
from dataclasses import replace
from io import StringIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import jax
from matplotlib import pyplot as plt
import numpy as np
import optax

from disco_rl import utils
from disco_rl.ppo import PPOConfig
from disco_rl.ppo import train_ppo
from disco_rl.ppo.meta_training import MetaTrainingConfig
from disco_rl.ppo.meta_training import MetaTrainingTrainer
from disco_rl.ppo.meta_training import _initial_student
from disco_rl.ppo.meta_training import save_meta_results
from disco_rl.ppo.meta_training import train_meta
from disco_rl.ppo.test_imitation import small_agent_settings
from disco_rl.ppo.test_ppo import get_env


class MetaTrainingTest(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    with redirect_stdout(StringIO()):
      cls.teacher = train_ppo(get_env, PPOConfig(
          seeds=(302,), num_updates=1, num_envs=2, horizon=8,
          epochs=1, num_minibatches=1, eval_every=1,
          eval_num_envs=2, eval_steps=14, checkpoint_steps=(),
      ))
    cls.settings = small_agent_settings()
    cls.config = MetaTrainingConfig(
        seeds=(511, 512), num_updates=2, num_inner_steps=2,
        num_envs=2, rollout_len=8, eval_every=1,
        eval_num_envs=2, eval_steps=14, fresh_seeds=(521,),
        fresh_updates=2,
    )

  def test_meta_gradient_requires_inner_updates(self):
    trainer = MetaTrainingTrainer(self.teacher, self.settings, self.config)
    learner, actor, ts, env_state, key = _initial_student(trainer, 511)
    rollouts = []
    for _ in range(self.config.num_inner_steps):
      key, subkey = jax.random.split(key)
      rollout, actor, ts, env_state = trainer.collect(
          learner.params, actor, ts, env_state, subkey
      )
      rollouts.append(rollout)
    validation, _, _, _ = trainer.collect_validation(
        learner.params, actor, ts, env_state, key
    )
    train_rollouts = utils.tree_stack(rollouts)
    eta, _ = trainer.agent.update_rule.init_params(jax.random.PRNGKey(501))
    (loss, (updated, metrics)), grads = trainer.gradient(
        eta, learner, train_rollouts, validation, key
    )
    self.assertTrue(np.isfinite(float(loss)))
    self.assertGreater(float(optax.global_norm(grads)), 0)
    self.assertGreater(float(metrics['agent_update_norm']), 0)
    self.assertGreater(float(optax.global_norm(jax.tree.map(
        lambda a, b: a - b, updated.params, learner.params
    ))), 0)
    self.assertTrue(all(np.isfinite(x).all() for x in jax.tree.leaves(grads)))
    self.assertEqual(
        [trainer.agent.settings.hyper_params[name] for name in (
            'y_cost', 'z_cost', 'aux_policy_cost', 'value_cost'
        )], [0, 0, 0, 0],
    )
    # Removing agent updates must remove eta's entire route to the outer loss.
    with patch.object(trainer.agent, '_optimizer', optax.set_to_zero()):
      frozen = learner.replace(opt_state=trainer.agent._optimizer.init(learner.params))
      (_, _), zero_grads = jax.value_and_grad(
          trainer.outer_loss, has_aux=True
      )(eta, frozen, train_rollouts, validation, key)
    self.assertEqual(float(optax.global_norm(zero_grads)), 0)
    # Deployment of a learned rule must not query PPO.
    with patch.object(
        trainer.student, 'teacher_samples',
        side_effect=AssertionError('PPO must not supply inner targets.'),
    ):
      fresh, _, _ = trainer.apply_rule(
          eta, learner, rollouts[0].first_state(time_axis=0), rollouts[0], key
      )
    self.assertGreater(float(optax.global_norm(jax.tree.map(
        lambda a, b: a - b, fresh.params, learner.params
    ))), 0)

  def test_training_updates_eta_freezes_teacher_and_saves(self):
    before = jax.tree.map(lambda x: np.array(x, copy=True), self.teacher.final_params)
    with redirect_stdout(StringIO()):
      result = train_meta(self.teacher, self.settings, self.config)
    self.assertEqual(len(result.training), 2)
    self.assertEqual(len(result.evaluation), 12)
    self.assertEqual(len(result.fresh_evaluation), 6)
    self.assertEqual(result.training.env_steps.tolist(), [128, 256])
    self.assertEqual(sorted(result.final_params), [511, 512])
    self.assertTrue((result.training.meta_gradient_norm > 0).all())
    self.assertTrue((result.training.meta_update_norm > 0).all())
    self.assertGreater(float(optax.global_norm(jax.tree.map(
        lambda a, b: a - b, result.final_meta_params, result.initial_meta_params
    ))), 0)
    for frame in (result.training, result.evaluation, result.fresh_evaluation):
      self.assertTrue(np.isfinite(frame.select_dtypes('number')).all().all())
    for old, new in zip(jax.tree.leaves(before), jax.tree.leaves(self.teacher.final_params)):
      np.testing.assert_array_equal(old, new)
    self.assertEqual(self.settings.hyper_params.y_cost, 1)
    with tempfile.TemporaryDirectory() as directory:
      with redirect_stdout(StringIO()):
        fig = save_meta_results(result, Path(directory))
      plt.close(fig)
      for filename in (
          'training.csv', 'evaluation.csv', 'fresh_evaluation.csv',
          'learning_curves.png',
      ):
        self.assertGreater((Path(directory) / filename).stat().st_size, 0)

  def test_full_inner_losses_preserve_initialisation_and_outer_objective(self):
    config = replace(
        self.config, use_auxiliary_losses=True, seeds=(511,), num_updates=1,
        fresh_updates=1,
    )
    full = MetaTrainingTrainer(self.teacher, self.settings, config)
    policy = MetaTrainingTrainer(self.teacher, self.settings, replace(
        config, use_auxiliary_losses=False
    ))
    for cost in ('y_cost', 'z_cost', 'aux_policy_cost', 'value_cost'):
      self.assertEqual(full.agent.settings.hyper_params[cost], self.settings.hyper_params[cost])
      self.assertEqual(policy.agent.settings.hyper_params[cost], 0)
    initial = _initial_student(full, 511)
    baseline_initial = _initial_student(policy, 511)
    for a, b in zip(jax.tree.leaves(initial), jax.tree.leaves(baseline_initial)):
      np.testing.assert_array_equal(a, b)
    learner, actor, ts, env_state, key = initial
    rollout, _, _, _ = full.collect(learner.params, actor, ts, env_state, key)
    targets = full.student.teacher_samples(rollout.observations['observation'][:-1])
    for a, b in zip(
        jax.tree.leaves(full.student.policy_loss(learner.params, actor, rollout, targets)),
        jax.tree.leaves(policy.student.policy_loss(learner.params, actor, rollout, targets)),
    ):
      np.testing.assert_array_equal(a, b)
    eta, _ = full.agent.update_rule.init_params(jax.random.PRNGKey(config.meta_seed))
    with patch.object(
        full.student, 'teacher_samples',
        side_effect=AssertionError('PPO must not supply inner targets.'),
    ):
      updated, _, metrics = full.apply_rule(eta, learner, actor, rollout, key)
    for name in ('y_loss', 'z_loss', 'aux_kl_loss', 'q_loss'):
      self.assertTrue(np.isfinite(float(metrics[name])))
      self.assertGreater(float(metrics[name]), 0)
    # Each restored objective must actually contribute gradients to the agent.
    for cost in ('y_cost', 'z_cost', 'aux_policy_cost', 'value_cost'):
      with self.subTest(cost=cost):
        settings = full.agent.settings.hyper_params
        original = settings.to_dict()
        try:
          for name in ('pi_cost', 'y_cost', 'z_cost', 'aux_policy_cost', 'value_cost'):
            settings[name] = original[name] if name == cost else 0
          grads = jax.grad(full.agent._loss, has_aux=True)(
              learner.params, actor, learner.meta_state, rollout,
              metrics['meta_out'], True,
          )[0]
        finally:
          settings.update(original)
        self.assertGreater(float(optax.global_norm(grads)), 0)
    self.assertGreater(float(optax.global_norm(jax.tree.map(
        lambda a, b: a - b, updated.params, learner.params
    ))), 0)
    teacher_before = jax.tree.map(
        lambda x: np.array(x, copy=True), self.teacher.final_params
    )
    with redirect_stdout(StringIO()):
      result = train_meta(self.teacher, self.settings, config)
    self.assertGreater(result.training.meta_gradient_norm.iloc[0], 0)
    self.assertGreater(result.training.meta_update_norm.iloc[0], 0)
    for name in ('y_loss', 'z_loss', 'aux_kl_loss', 'q_loss'):
      self.assertGreater(result.training['inner_' + name].iloc[0], 0)
    self.assertEqual(len(result.fresh_evaluation), 4)
    for a, b in zip(jax.tree.leaves(teacher_before), jax.tree.leaves(self.teacher.final_params)):
      np.testing.assert_array_equal(a, b)
    self.assertEqual(self.settings.hyper_params.y_cost, 1)

  def test_full_inner_losses_reject_disabled_weights(self):
    settings = small_agent_settings()
    settings.hyper_params.value_cost = 0
    with self.assertRaisesRegex(ValueError, 'value_cost'):
      MetaTrainingTrainer(self.teacher, settings, replace(
          self.config, use_auxiliary_losses=True
      ))


if __name__ == '__main__':
  unittest.main()
