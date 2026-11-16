"""Check signed steps, surrogate-gradient comparisons, and fixed replay."""

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import tempfile
import unittest

import jax
import jax.numpy as jnp
from matplotlib import pyplot as plt
import numpy as np
import optax

from disco_rl.ppo import PPOConfig
from disco_rl.ppo import train_ppo
from disco_rl.ppo.meta_training import MetaTrainingConfig
from disco_rl.ppo.meta_training import MetaTrainingTrainer
from disco_rl.ppo.meta_update_validation import _measure_updates
from disco_rl.ppo.meta_update_validation import _norm
from disco_rl.ppo.meta_update_validation import probe_meta_updates
from disco_rl.ppo.meta_update_validation import save_meta_update_probe
from disco_rl.ppo.test_imitation import small_agent_settings
from disco_rl.ppo.test_ppo import get_env


class MetaUpdateValidationTest(unittest.TestCase):

  def tearDown(self):
    plt.close('all')

  def test_signed_steps_and_exact_directional_derivatives(self):
    params = {'x': jnp.array([1.0, -0.5])}
    fn = jax.jit(lambda p: (p['x']**2).sum())
    grads = jax.grad(fn)(params)
    result = _measure_updates(
        fn, params, grads, optax.adam(1e-3), (1e-3, 1e-2), 1e-7
    )
    self.assertEqual(result.replay_error, 0)
    self.assertEqual(len(result.trials), 9)
    self.assertTrue(result.trials.loc[result.trials.sign == 1, 'resolved_decrease'].all())
    self.assertTrue(result.trials.loc[result.trials.sign == -1, 'resolved_increase'].all())
    np.testing.assert_allclose(
        result.slopes.measured_central_slope, result.slopes.predicted_slope,
        rtol=2e-4, atol=2e-4,
    )
    np.testing.assert_array_equal(params['x'], [1.0, -0.5])
    with tempfile.TemporaryDirectory() as directory:
      with redirect_stdout(StringIO()):
        save_meta_update_probe(result, Path(directory))
      for filename in ('trials.csv', 'directional_slopes.csv', 'summary.csv', 'loss_change.png'):
        self.assertGreater((Path(directory) / filename).stat().st_size, 0)

  def test_stop_gradient_mismatch_is_measured_not_hidden(self):
    params = {'x': jnp.array([0.5, 1.0])}

    def fn(p):
      x = p['x']
      surrogate = x + jax.lax.stop_gradient(x)
      return (surrogate**2).sum()

    result = _measure_updates(
        jax.jit(fn), params, jax.grad(fn)(params), optax.adam(1e-3),
        (1e-3,), 1e-7,
    )
    np.testing.assert_allclose(
        result.slopes.measured_central_slope, 2 * result.slopes.predicted_slope,
        rtol=1e-3,
    )

  def test_historical_adam_direction_preserves_state_and_exposes_momentum(self):
    optimizer = optax.adam(0.1)
    params = {'x': jnp.array([1.0])}
    state = optimizer.init(params)
    _, state = optimizer.update({'x': jnp.array([-10.0])}, state, params)
    original = jax.tree.map(lambda x: np.array(x, copy=True), state)
    fn = jax.jit(lambda p: (p['x']**2).sum())
    result = _measure_updates(
        fn, params, jax.grad(fn)(params), optimizer, (1e-3, 1e-2), 1e-7,
        meta_opt_state=state,
    )
    self.assertEqual(len(result.trials), 13)
    self.assertGreater(result.historical_adam_update_norm, 0)
    forward = result.trials[result.trials.sign == 1]
    self.assertTrue(forward.loc[forward.direction == 'historical_adam', 'resolved_increase'].all())
    self.assertTrue(forward.loc[forward.direction == 'negative_gradient', 'resolved_decrease'].all())
    np.testing.assert_allclose(
        result.slopes.measured_central_slope, result.slopes.predicted_slope,
        rtol=2e-4, atol=2e-4,
    )
    for before, after in zip(jax.tree.leaves(original), jax.tree.leaves(state)):
      np.testing.assert_array_equal(before, after)

  def test_fixed_disco_replay_leaves_teacher_and_eta_unchanged(self):
    with redirect_stdout(StringIO()):
      teacher = train_ppo(get_env, PPOConfig(
          seeds=(302,), num_updates=1, num_envs=1, horizon=8,
          epochs=1, num_minibatches=1, eval_every=1,
          eval_num_envs=1, eval_steps=7, checkpoint_steps=(),
      ))
    settings = small_agent_settings()
    config = MetaTrainingConfig(
        seeds=(811,), num_updates=1, num_inner_steps=1, num_envs=1,
        rollout_len=8, eval_num_envs=1, eval_steps=7, fresh_seeds=(812,),
        fresh_updates=1, use_auxiliary_losses=True,
    )
    trainer = MetaTrainingTrainer(teacher, settings, config)
    eta, _ = trainer.agent.update_rule.init_params(jax.random.PRNGKey(501))
    original = jax.tree.map(lambda x: np.array(x, copy=True), (eta, teacher.final_params))
    with redirect_stdout(StringIO()):
      result = probe_meta_updates(
          teacher, settings, config, meta_params=eta, step_norms=(1e-3, 1e-2)
      )
    self.assertEqual(result.replay_error, 0)
    self.assertGreater(result.gradient_norm, 0)
    self.assertEqual(len(result.trials), 9)
    self.assertTrue(np.isfinite(result.trials.select_dtypes('number')).all().all())
    for old, current in zip(
        jax.tree.leaves(original), jax.tree.leaves((eta, teacher.final_params))
    ):
      np.testing.assert_array_equal(old, current)
    self.assertEqual(settings.hyper_params.y_cost, 1)

  def test_invalid_steps(self):
    fn = lambda p: p['x'].sum()
    params = {'x': jnp.ones(2)}
    with self.assertRaises(ValueError):
      _measure_updates(fn, params, params, optax.adam(1e-3), (0,), 1e-7)

  def test_small_gradient_norm_does_not_underflow(self):
    params = {'x': jnp.ones(2)}
    grads = {'x': jnp.full(2, 1e-30)}
    self.assertGreater(_norm(grads), 0)
    np.testing.assert_allclose(_norm(grads), np.sqrt(2) * 1e-30, rtol=1e-6)
    result = _measure_updates(
        lambda p: p['x'].sum(), params, grads, optax.adam(1e-3),
        (1e-3,), 1e-7,
    )
    self.assertEqual(result.gradient_nonzero_count, 2)
    self.assertEqual(set(result.trials.direction), {'baseline', 'negative_gradient', 'fresh_adam'})

  def test_zero_gradient_reports_unavailable_and_tests_loss_sensitivity(self):
    params = {'x': jnp.ones(2)}
    fn = jax.jit(lambda p: jax.lax.stop_gradient(p['x']).sum())
    optimizer = optax.adam(1e-3)
    with redirect_stdout(StringIO()) as output:
      result = _measure_updates(
          fn, params, jax.grad(fn)(params), optimizer, (1e-2,), 1e-7,
          meta_opt_state=optimizer.init(params),
      )
    self.assertIn('not convergence', output.getvalue())
    self.assertEqual(result.gradient_norm, 0)
    self.assertEqual(result.gradient_nonzero_count, 0)
    unavailable = result.direction_status[~result.direction_status.tested]
    self.assertEqual(set(unavailable.direction), {
        'negative_gradient', 'fresh_adam', 'historical_adam',
    })
    self.assertEqual(set(result.slopes.direction), {'random_control_0', 'random_control_1'})
    self.assertTrue(result.slopes.slope_resolved.all())
    np.testing.assert_array_equal(result.slopes.predicted_slope, [0, 0])
    self.assertTrue(np.isfinite(result.trials.select_dtypes('number')).all().all())
    with tempfile.TemporaryDirectory() as directory:
      with redirect_stdout(StringIO()):
        save_meta_update_probe(result, Path(directory))
      self.assertGreater((Path(directory) / 'direction_status.csv').stat().st_size, 0)

  def test_zero_gradient_still_tests_historical_momentum_and_flat_controls(self):
    params = {'x': jnp.ones(2)}
    optimizer = optax.adam(0.1)
    _, state = optimizer.update({'x': jnp.ones(2)}, optimizer.init(params), params)
    with redirect_stdout(StringIO()):
      result = _measure_updates(
          lambda p: jnp.array(1.0), params, {'x': jnp.zeros(2)}, optimizer,
          (1e-2,), 1e-7, meta_opt_state=state,
      )
    self.assertGreater(result.historical_adam_update_norm, 0)
    self.assertIn('historical_adam', set(result.slopes.direction))
    self.assertFalse(result.slopes.slope_resolved.any())


if __name__ == '__main__':
  unittest.main()
