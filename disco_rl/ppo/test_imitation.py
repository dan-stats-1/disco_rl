"""Policy-only gradient, teacher-freezing, and reward-accounting tests."""

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import jax
import jax.numpy as jnp
from matplotlib import pyplot as plt
import numpy as np

from disco_rl import agent as agent_lib
from disco_rl.ppo import PPOConfig
from disco_rl.ppo import train_ppo
from disco_rl.ppo.imitation import ImitationConfig
from disco_rl.ppo.imitation import ImitationTrainer
from disco_rl.ppo.imitation import save_imitation_results
from disco_rl.ppo.imitation import train_imitation
from disco_rl.ppo.test_ppo import get_env


def small_agent_settings():
  settings = agent_lib.get_settings_disco(
      continuous_actions=True, num_samples=8
  )
  settings.net_settings.net_args = dict(
      dense=(16, 16), model_arch_name='lstm', head_w_init_std=1e-2,
      model_kwargs=dict(head_mlp_hiddens=(8,), lstm_size=8),
  )
  settings.update_rule.net.prediction_size = 8
  settings.update_rule.net.hidden_size = 8
  settings.update_rule.net.meta_rnn_kwargs.hidden_size = 8
  return settings


class ImitationTest(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    with redirect_stdout(StringIO()):
      cls.teacher = train_ppo(get_env, PPOConfig(
          seeds=(302,), num_updates=2, num_envs=2, horizon=16,
          epochs=1, num_minibatches=2, eval_every=1,
          eval_num_envs=2, eval_steps=16, checkpoint_steps=(32,),
      ))
    cls.settings = small_agent_settings()
    cls.config = ImitationConfig(
        seeds=(401,), num_updates=2, num_envs=2, rollout_len=8,
        updates_per_rollout=2, eval_every=1, eval_num_envs=2, eval_steps=16,
    )

  def test_quantile_targets_are_fixed_and_detached(self):
    trainer = ImitationTrainer(self.teacher, self.settings, self.config)
    obs = jnp.ones((3, 2, 3), dtype=jnp.float32)
    targets = trainer.teacher_samples(obs)
    self.assertEqual(targets.shape, (3, 2, 8))
    np.testing.assert_array_equal(targets, trainer.teacher_samples(obs))
    self.assertTrue((np.diff(targets, axis=-1) >= 0).all())
    self.assertTrue((np.abs(targets) <= 1).all())
    gradients = jax.grad(lambda x: trainer.teacher_samples(x).sum())(obs)
    np.testing.assert_array_equal(gradients, np.zeros(obs.shape))

  def test_only_policy_loss_and_no_meta_updates(self):
    trainer = ImitationTrainer(self.teacher, self.settings, self.config)
    self.assertEqual(
        [trainer.agent.settings.hyper_params[cost] for cost in (
            'y_cost', 'z_cost', 'aux_policy_cost', 'value_cost'
        )], [0, 0, 0, 0],
    )
    self.assertEqual(self.settings.hyper_params.y_cost, 1.0)
    keys = jax.random.split(jax.random.PRNGKey(401), 4)
    learner = trainer.agent.initial_learner_state(keys[0])
    actor_start = trainer.agent.initial_actor_state(keys[1])
    state, timestep = trainer.env.reset(keys[2])
    rollout, _, _, _ = trainer.collect(
        learner.params, actor_start, timestep, state, keys[3]
    )
    original_unroll = trainer.agent.unroll_net

    def unusable_auxiliary_outputs(*args):
      out, state = original_unroll(*args)
      out = {
          name: value if name == 'logits' else jnp.full_like(value, jnp.nan)
          for name, value in out.items()
      }
      return out, state

    targets = trainer.teacher_samples(rollout.observations['observation'][:-1])
    with patch.object(trainer.agent, 'unroll_net', unusable_auxiliary_outputs):
      loss, _ = trainer.policy_loss(
          learner.params, actor_start, rollout, targets
      )
    self.assertTrue(np.isfinite(float(loss)))
    with patch.object(
        trainer.agent.update_rule, 'unroll_meta_net',
        side_effect=AssertionError('The meta-network must not run.'),
    ):
      updated, _, metrics = trainer.learn(learner, actor_start, rollout)
    self.assertLess(float(metrics['pi_loss']), float(metrics['pi_loss_before']))
    self.assertGreater(float(metrics['gradient_norm']), 0)
    self.assertGreater(float(metrics['update_norm']), 0)
    for old, new in zip(
        jax.tree.leaves(learner.meta_state), jax.tree.leaves(updated.meta_state)
    ):
      np.testing.assert_array_equal(old, new)

  def test_teacher_unchanged_and_results_saved(self):
    teacher_snapshot = jax.tree.map(
        lambda x: np.array(x, copy=True), self.teacher.final_params
    )
    with redirect_stdout(StringIO()):
      result = train_imitation(self.teacher, self.settings, self.config)
    self.assertEqual(len(result.training), 2)
    self.assertEqual(len(result.evaluation), 6)
    self.assertEqual(len(result.teacher_evaluation), 2)
    self.assertEqual(result.training.env_steps.tolist(), [16, 32])
    self.assertEqual(result.training.optimiser_steps.tolist(), [2, 4])
    self.assertEqual(int(result.training.episodes.sum()), 4)
    self.assertTrue((result.evaluation.episodes == 4).all())
    np.testing.assert_allclose(
        result.evaluation.mean_return, 2 * result.evaluation.catch_rate - 1
    )
    self.assertTrue(np.isfinite(result.training.select_dtypes('number')).all().all())
    for before, after in zip(
        jax.tree.leaves(teacher_snapshot), jax.tree.leaves(self.teacher.final_params)
    ):
      np.testing.assert_array_equal(before, after)
    with tempfile.TemporaryDirectory() as directory:
      with redirect_stdout(StringIO()):
        fig = save_imitation_results(result, Path(directory))
      plt.close(fig)
      for name in (
          'training.csv', 'evaluation.csv', 'teacher_evaluation.csv',
          'learning_curves.png',
      ):
        self.assertGreater((Path(directory) / name).stat().st_size, 0)


if __name__ == '__main__':
  unittest.main()
