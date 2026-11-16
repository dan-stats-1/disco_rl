"""Tests of the KDE validation measurements and controlled learning setup."""

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import tempfile
import unittest

import jax.numpy as jnp
from matplotlib import pyplot as plt
import numpy as np

from disco_rl.kde_validation import _return_and_gradient
from disco_rl.kde_validation import _true_log_density
from disco_rl.kde_validation import save_validation
from disco_rl.kde_validation import validate_accuracy
from disco_rl.kde_validation import validate_learning
from disco_rl.kde_validation import validate_numerics


class KDEValidationTest(unittest.TestCase):

  def tearDown(self):
    plt.close('all')

  def test_numerics_and_saved_output(self):
    with redirect_stdout(StringIO()):
      result = validate_numerics()
    checks = result.tables['checks']
    self.assertTrue(checks.passed.all(), checks.loc[~checks.passed].to_string())
    self.assertIn('second derivative directional check', checks['check'].tolist())
    self.assertIn('outside_centres', checks.case.tolist())
    with tempfile.TemporaryDirectory() as directory:
      with redirect_stdout(StringIO()):
        save_validation(result, Path(directory))
      for name in ('checks.csv', 'validation.png'):
        self.assertGreater((Path(directory) / name).stat().st_size, 0)

  def test_exact_density_and_statistical_measurements(self):
    grid = np.linspace(-1, 1, 8193)
    p = np.exp(_true_log_density(grid, 0, 0.5))
    self.assertAlmostEqual(np.trapezoid(p, grid), 1, places=5)
    with redirect_stdout(StringIO()):
      result = validate_accuracy(
          sample_sizes=(128,), bandwidths=(0.03, 0.3),
          seeds=(601,), grid_size=1025, num_queries=128,
      )
    table = result.tables['accuracy']
    self.assertEqual(len(table), 6)
    self.assertEqual(set(table.case), {'central', 'narrow', 'near_boundary'})
    self.assertTrue(np.isfinite(table.select_dtypes('number')).all().all())
    self.assertTrue((abs(table.kde_mass - 1) < 0.01).all())
    narrow = table[table.case == 'narrow'].set_index('bandwidth')
    self.assertGreater(narrow.loc[0.3, 'density_l1'], narrow.loc[0.03, 'density_l1'])
    self.assertTrue((table.ratio_log_rmse >= 0).all())
    self.assertTrue((table.clipped_ratio_mae <= 1).all())

  def test_score_direction_and_actual_learning(self):
    expected, gradient = _return_and_gradient(0.6)
    point = jnp.array([0.0, -0.5])
    g = gradient(point)
    numerical = jnp.array([
        (expected(point + jnp.eye(2)[i] * 1e-3)
         - expected(point - jnp.eye(2)[i] * 1e-3)) / 2e-3
        for i in range(2)
    ])
    np.testing.assert_allclose(g, numerical, atol=2e-4, rtol=2e-3)
    self.assertGreater(float(g[0]), 0)
    self.assertLess(float(g[1]), 0)
    with redirect_stdout(StringIO()):
      result = validate_learning(
          sample_sizes=(32,), bandwidths=(0.1,), seeds=(701, 702),
          action_batches=128, learning_samples=32, num_updates=25,
          learning_rate=0.15,
      )
    probes, learning = result.tables['gradients'], result.tables['learning']
    self.assertEqual(set(probes.estimator), {
        'exact', 'kde_self_bank', 'kde_independent_bank'
    })
    self.assertTrue((probes.loc[probes.estimator == 'exact', 'cosine'] > 0.95).all())
    self.assertTrue(np.isfinite(probes.select_dtypes('number')).all().all())
    self.assertEqual(len(learning), 2 * 3 * 26)
    for _, group in learning[learning.estimator == 'exact'].groupby('seed'):
      self.assertGreater(group.expected_return.iloc[-1] - group.expected_return.iloc[0], 0.1)
    self.assertTrue(np.isfinite(learning.select_dtypes('number')).all().all())

  def test_invalid_configuration(self):
    with self.assertRaises(ValueError):
      validate_numerics(bandwidths=(0,))
    with self.assertRaises(ValueError):
      validate_accuracy(sample_sizes=(1,))
    with self.assertRaises(ValueError):
      validate_learning(target=1)


if __name__ == '__main__':
  unittest.main()
