"""Verify fixed-loss optimisation, held-out isolation and real Disco replay."""

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from unittest.mock import Mock

import jax
import jax.numpy as jnp
from matplotlib import pyplot as plt
import numpy as np
import optax

from disco_rl.ppo import PPOConfig
from disco_rl.ppo import train_ppo
from disco_rl.ppo.fixed_meta_training import _optimise_fixed_loss
from disco_rl.ppo.fixed_meta_training import fit_fixed_meta
from disco_rl.ppo.fixed_meta_training import probe_fixed_meta_updates
from disco_rl.ppo.fixed_meta_training import save_fixed_meta_results
from disco_rl.ppo.meta_training import MetaTrainingConfig
from disco_rl.ppo.meta_update_validation import _fixed_snapshot
from disco_rl.ppo.test_imitation import small_agent_settings
from disco_rl.ppo.test_ppo import get_env


class FixedMetaTrainingTest(unittest.TestCase):

  def tearDown(self):
    plt.close('all')

  def test_outer_history_persists_and_heldout_is_not_optimised(self):
    params = {'x': jnp.array([0.0])}

    def losses(p):
      return ((p['x'] - 1)**2).sum(), ((p['x'] + 1)**2).sum()

    with redirect_stdout(StringIO()):
      result = _optimise_fixed_loss(
          params, losses, optax.adam(0.1), 4, (0.01, 0.1, 1), 1, 1e-7, 1
      )
    self.assertEqual(len(result.training), 5)
    self.assertEqual(len(result.step_trials), 16)
    self.assertEqual(int(result.final_meta_opt_state[0].count), 4)
    self.assertLess(result.training.fit_loss.iloc[-1], result.training.fit_loss.iloc[0])
    self.assertGreater(result.training.heldout_loss.iloc[-1], result.training.heldout_loss.iloc[0])
    self.assertEqual(result.replay_error, 0)
    np.testing.assert_array_equal(params['x'], [0])
    zeros = result.step_trials[result.step_trials.multiplier == 0]
    np.testing.assert_array_equal(zeros.fit_loss_change, np.zeros(4))
    np.testing.assert_array_equal(zeros.heldout_loss_change, np.zeros(4))
    committed = result.step_trials[result.step_trials.committed]
    np.testing.assert_allclose(committed.fit_loss, result.training.fit_loss.iloc[1:])
    with tempfile.TemporaryDirectory() as directory:
      with redirect_stdout(StringIO()):
        save_fixed_meta_results(result, Path(directory))
      for filename in ('training.csv', 'step_trials.csv', 'summary.csv', 'learning_curves.png'):
        self.assertGreater((Path(directory) / filename).stat().st_size, 0)

  def test_trials_do_not_automatically_select_the_best_step(self):
    params = {'x': jnp.array([1.0])}
    losses = lambda p: ((p['x']**2).sum(), (p['x']**2).sum())
    with redirect_stdout(StringIO()):
      result = _optimise_fixed_loss(
          params, losses, optax.sgd(1.5), 1, (0.1, 1), 1, 1e-7, 1
      )
    self.assertTrue(result.step_trials.loc[result.step_trials.multiplier == 0.1, 'resolved_decrease'].all())
    self.assertTrue(result.training.resolved_increase.iloc[-1])
    np.testing.assert_allclose(result.final_meta_params['x'], [-2])
    self.assertEqual(result.training.fit_loss.iloc[-1], 4)

  def test_real_fixed_snapshot_is_unchanged(self):
    with redirect_stdout(StringIO()):
      teacher = train_ppo(get_env, PPOConfig(
          seeds=(302,), num_updates=1, num_envs=1, horizon=8,
          epochs=1, num_minibatches=1, eval_every=1, eval_num_envs=1,
          eval_steps=7, checkpoint_steps=(),
      ))
    teacher_before = jax.tree.map(lambda x: np.array(x, copy=True), teacher.final_params)
    settings = small_agent_settings()
    config = MetaTrainingConfig(
        seeds=(811,), num_updates=2, num_inner_steps=1, num_envs=1,
        rollout_len=8, eval_num_envs=1, eval_steps=7, fresh_seeds=(812,),
        fresh_updates=1,
    )
    captured = []

    def capture(*args):
      snapshot = _fixed_snapshot(*args)
      captured.append((snapshot[1:], jax.tree.map(
          lambda x: np.array(x, copy=True), snapshot[1:]
      )))
      return snapshot

    with patch('disco_rl.ppo.fixed_meta_training._fixed_snapshot', side_effect=capture):
      with redirect_stdout(StringIO()):
        result = fit_fixed_meta(
            teacher, settings, config, step_multipliers=(0.1, 1), log_every=1
        )
    self.assertEqual(len(captured), 1)
    self.assertEqual(len(result.training), 3)
    self.assertEqual(len(result.step_trials), 6)
    self.assertEqual(result.replay_error, 0)
    self.assertTrue(np.isfinite(result.training.select_dtypes('number')).all().all())
    self.assertTrue((result.training.gradient_norm_before.iloc[1:] > 0).all())
    self.assertTrue((result.training.applied_step_norm.iloc[1:] > 0).all())
    snapshot, before = captured[0]
    for old, current in zip(jax.tree.leaves(before), jax.tree.leaves(snapshot)):
      np.testing.assert_array_equal(old, current)
    for old, current in zip(jax.tree.leaves(teacher_before), jax.tree.leaves(teacher.final_params)):
      np.testing.assert_array_equal(old, current)
    self.assertEqual(settings.hyper_params.y_cost, 1)
    checkpoint = jax.tree.map(
        lambda x: np.array(x, copy=True),
        (result.final_meta_params, result.final_meta_opt_state),
    )
    with redirect_stdout(StringIO()):
      probe = probe_fixed_meta_updates(
          teacher, settings, config, result, step_norms=(1e-3, 1e-2),
      )
    self.assertLessEqual(probe.reference_loss_error, 1e-6)
    self.assertEqual(len(probe.trials), 13)
    self.assertIn('historical_adam', set(probe.trials.direction))
    for before, after in zip(
        jax.tree.leaves(checkpoint),
        jax.tree.leaves((result.final_meta_params, result.final_meta_opt_state)),
    ):
      np.testing.assert_array_equal(before, after)
    result.training.loc[result.training.index[-1], 'fit_loss'] += 0.1
    with self.assertRaisesRegex(RuntimeError, 'Checkpoint replay loss does not match'):
      probe_fixed_meta_updates(
          teacher, settings, config, result, step_norms=(1e-3,),
      )

  def test_invalid_factors(self):
    params = {'x': jnp.ones(1)}
    with self.assertRaises(ValueError):
      _optimise_fixed_loss(
          params, lambda p: (p['x'].sum(), p['x'].sum()), optax.adam(1e-3),
          1, (-1,), 1, 1e-7, 1,
      )

  def test_reference_is_checked_before_initialising_outer_optimizer(self):
    params = {'x': jnp.ones(1)}
    adam = optax.adam(1e-3)
    init, update = Mock(wraps=adam.init), Mock(wraps=adam.update)
    optimizer = optax.GradientTransformation(init, update)
    with self.assertRaisesRegex(RuntimeError, 'Initial fixed-fit loss does not match'):
      _optimise_fixed_loss(
          params, lambda p: (p['x'].sum(), p['x'].sum()), optimizer,
          100, (0.1, 1), 1, 1e-7, 10, expected_initial_fit_loss=2.0,
      )
    init.assert_not_called()
    update.assert_not_called()
    with redirect_stdout(StringIO()):
      result = _optimise_fixed_loss(
          params, lambda p: (p['x'].sum(), p['x'].sum()), optimizer,
          1, (0.1, 1), 1, 1e-7, 1, expected_initial_fit_loss=1.0,
      )
    self.assertEqual(len(result.training), 2)


if __name__ == '__main__':
  unittest.main()
