"""Focused PPO and visualisation regression tests (python -m unittest)."""

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import tempfile
import unittest

import jax
import jax.numpy as jnp
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from disco_rl.environments import jittable_envs
from disco_rl.ppo import PPOConfig
from disco_rl.ppo import save_training_results
from disco_rl.ppo import train_ppo
from disco_rl.ppo import visualize_episode


def get_env(batch_size):
  return jittable_envs.ContinuousCatchJittableEnvironment(
      batch_size, jittable_envs.get_config_continuous_catch()
  )


class PPOTest(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    cls.config = PPOConfig(
        seeds=(301, 302), num_updates=2, num_envs=2, horizon=16,
        epochs=1, num_minibatches=2, eval_every=1,
        eval_num_envs=2, eval_steps=16, checkpoint_steps=(32,),
    )
    with redirect_stdout(StringIO()):
      cls.result = train_ppo(get_env, cls.config)

  def test_config_validation(self):
    with self.assertRaises(ValueError):
      PPOConfig(num_minibatches=0)
    with self.assertRaises(ValueError):
      PPOConfig(checkpoint_steps=(25_000,))

  def test_histories_and_completed_episodes(self):
    self.assertEqual(len(self.result.training), 4)
    self.assertEqual(len(self.result.evaluation), 14)
    self.assertTrue((self.result.evaluation.episodes == 4).all())
    self.assertTrue(self.result.evaluation.catch_rate.between(0, 1).all())
    np.testing.assert_allclose(
        self.result.evaluation.mean_return,
        2 * self.result.evaluation.catch_rate - 1,
    )
    self.assertTrue(np.isfinite(
        self.result.training.select_dtypes('number').to_numpy()
    ).all())

  def test_terminal_targets_and_action_bounds(self):
    trainer = self.result.trainer
    params, state, obs, key, _ = trainer.initial_state(301)
    _, _, _, batch, stats = trainer.collect(
        params, state, obs, jnp.zeros(2), key
    )
    returns = np.asarray(batch['returns']).reshape((16, 2))
    np.testing.assert_allclose(np.abs(returns[[6, 13]]), 1.0, atol=1e-6)
    self.assertEqual(int(stats[1]), 4)
    np.testing.assert_allclose(
        trainer.bounded_action(jnp.array([-100.0, 0.0, 100.0])),
        [-1.0, 0.0, 1.0],
    )
    with self.assertRaises(ValueError):
      trainer.evaluate(params, key, 'unknown')

  def test_replayed_checkpoint_matches_training(self):
    saved = self.result.checkpoints[301][32]
    replayed = self.result.trainer.replay_checkpoints(301, 32)[32]
    for old, new in zip(jax.tree.leaves(saved), jax.tree.leaves(replayed)):
      np.testing.assert_allclose(old, new, rtol=1e-6, atol=1e-7)
    with self.assertRaises(ValueError):
      self.result.checkpoint(301, 25)
    with self.assertRaises(ValueError):
      self.result.checkpoint(999, 0)

  def test_saved_plots_and_episode_animation(self):
    snapshot = jax.tree.map(
        lambda x: np.array(x, copy=True), self.result.final_params[301]
    )
    with tempfile.TemporaryDirectory() as directory:
      output_dir = Path(directory)
      with redirect_stdout(StringIO()):
        figure = save_training_results(self.result, output_dir)
        comparison = visualize_episode(
            self.result, training_seed=301, episode_seed=0,
            middle_steps=32, output_dir=output_dir,
        )
      plt.close(figure)
      for name in (
          'training.csv', 'evaluation.csv', 'learning_curves.png',
          'episode_comparison.csv', 'episode_comparison.html',
          'episode_comparison.gif', 'episode_comparison_final.png',
      ):
        self.assertGreater((output_dir / name).stat().st_size, 0)
      self.assertEqual(
          len(pd.read_csv(output_dir / 'evaluation.csv')),
          len(self.result.evaluation),
      )
      self.assertEqual(
          comparison.summary.training_transitions.tolist(), [0, 32, 64]
      )
      self.assertIn('animation', comparison.html)
      for episode in comparison.episodes:
        self.assertEqual(len(episode.frames), 8)
        self.assertEqual(float(episode.frames[-1, 0]), 7.0)
        np.testing.assert_array_equal(
            episode.frames[0], comparison.episodes[0].frames[0]
        )
        caught = abs(episode.frames[-1, 1] - episode.frames[-1, 3]) <= 0.5
        self.assertEqual(episode.returns[-1], 1.0 if caught else -1.0)
      with Image.open(output_dir / 'episode_comparison.gif') as image:
        self.assertEqual(image.format, 'GIF')
        self.assertEqual(image.n_frames, 8)
        self.assertEqual(image.info['duration'], 400)
        image.seek(image.n_frames - 1)
        self.assertEqual(image.info['duration'], 2000)
    for before, after in zip(
        jax.tree.leaves(snapshot),
        jax.tree.leaves(self.result.final_params[301]),
    ):
      np.testing.assert_array_equal(before, after)


if __name__ == '__main__':
  unittest.main()
