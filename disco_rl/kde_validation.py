"""Three independent KDE experiments, in increasing order of complexity.

1. ``validate_numerics``: is the bounded KDE formula and its autodiff correct?
2. ``validate_accuracy``: how well does a finite sample bank approximate a
   distribution whose density, entropy and probability ratios we know?
3. ``validate_learning``: do KDE policy gradients point towards higher reward,
   and can they train a small policy on a controlled one-step task?

A bank is a collection of action samples used to fit the density estimate.
The bandwidth is the width of the Gaussian kernel around each sample.
These checks do not train a Disco rule or establish that Catch/PPO will learn.
"""

from dataclasses import dataclass
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
from scipy import stats

from disco_rl import continuous_policies as policies


@dataclass
class ValidationResult:
  tables: dict[str, pd.DataFrame]
  figure: Figure


def save_validation(result: ValidationResult, output_dir: Path) -> None:
  output_dir.mkdir(parents=True, exist_ok=True)
  for name, table in result.tables.items():
    table.to_csv(output_dir / f'{name}.csv', index=False)
  result.figure.savefig(output_dir / 'validation.png', dpi=150)
  print(f'KDE validation tables and figure saved to {output_dir}')


def _settings(sample_sizes, bandwidths, seeds):
  if (
      not sample_sizes or not bandwidths or not seeds
      or any(not isinstance(n, int) or n < 2 for n in sample_sizes)
      or any(not np.isfinite(h) or h <= 0 for h in bandwidths)
  ):
    raise ValueError('Use sample sizes >= 2, positive bandwidths, and nonempty seeds.')


def _finite(table: pd.DataFrame) -> pd.DataFrame:
  if not np.isfinite(table.select_dtypes('number').to_numpy()).all():
    raise FloatingPointError('Non-finite KDE validation measurements.')
  return table


def _density_grid(bank, grid, bandwidth):
  # vmap exercises the public scalar-query API without materialising Q copies.
  return jax.vmap(lambda a: policies.kde_log_prob(bank, a, bandwidth))(grid)


def _finite_difference(fn, point, step=1e-3):
  """Estimate each derivative by slightly increasing/decreasing one input."""
  point = np.asarray(point, dtype=np.float32)
  result = np.empty_like(point)
  for index in np.ndindex(point.shape):
    offset = np.zeros_like(point)
    offset[index] = step
    result[index] = (
        float(fn(jnp.asarray(point + offset)))
        - float(fn(jnp.asarray(point - offset)))
    ) / (2 * step)
  return result


def _check_density(bank_np, grid, bandwidth, record):
  """Check normalisation, an independent formula, bounds and sample ordering."""
  bank = jnp.asarray(bank_np)
  log_density = np.asarray(_density_grid(bank, grid, bandwidth))

  # Check 1: a probability density must integrate to one on the action interval.
  integral = np.trapezoid(np.exp(log_density), np.asarray(grid))
  record('density integrates to one', abs(integral - 1), 3e-3)

  # Check 2: compare JAX's KDE with SciPy's independently implemented kernels.
  queries = np.array([-1, -0.99, -0.5, 0, 0.5, 0.99, 1], np.float32)
  centres = bank_np.astype(np.float64)
  reference = stats.truncnorm.pdf(
      queries.astype(np.float64)[:, None],
      (-1 - centres) / bandwidth, (1 - centres) / bandwidth,
      loc=centres, scale=bandwidth,
  ).mean(axis=-1)
  observed = np.exp(np.asarray(_density_grid(bank, jnp.asarray(queries), bandwidth)))
  record('independent truncated-normal reference',
         np.max(np.abs(observed - reference) / (1 + reference)), 2e-5)

  # Check 3: impossible actions must have log probability -inf (density zero).
  outside = _density_grid(bank, jnp.array([-1.01, 1.01]), bandwidth)
  record('zero density outside bounds',
         0 if bool(jnp.isneginf(outside).all()) else 1, 0)

  # Check 4: reordering the same samples must not change the estimated density.
  reversed_log_density = np.asarray(_density_grid(bank[::-1], grid, bandwidth))
  record('bank permutation invariance',
         np.max(abs(log_density - reversed_log_density)), 2e-5)
  return np.exp(log_density)


def _check_first_derivatives(bank, action, bandwidth, record):
  """Compare autodiff with finite differences of the same scalar objectives."""
  objectives = {
      'log-density sample gradient': lambda b: policies.kde_log_prob(b, action, bandwidth),
      'entropy sample/query gradient': lambda b: policies.kde_entropy(b, bandwidth),
      'policy-score sample gradient': lambda b: policies.kde_policy_gradient_loss(
          b, action, jnp.array(0.7), bandwidth
      ),
  }

  # Check 5: moving bank samples should give the derivatives predicted by JAX.
  for name, objective in objectives.items():
    automatic = np.asarray(jax.grad(objective)(bank))
    numerical = _finite_difference(objective, bank)
    error = np.max(abs(automatic - numerical)) / (1 + np.max(abs(numerical)))
    record(name, error, 5e-3)

  # Check 6: also check the derivative when the density query itself moves.
  def query_log_density(query):
    return policies.kde_log_prob(bank, query, bandwidth)

  automatic = float(jax.grad(query_log_density)(action))
  numerical = float(_finite_difference(query_log_density, action))
  record('log-density query gradient',
         abs(automatic - numerical) / (1 + abs(numerical)), 5e-3)
  return objectives['policy-score sample gradient']


def _check_second_derivative(bank, policy_score_loss, record):
  """Check a derivative of the gradient, needed for differentiating learning."""
  # Check 7: compare a Hessian-vector product with changes in the first gradient.
  direction = jnp.linspace(-0.2, 0.2, bank.size)
  hessian_direction = jax.jvp(jax.grad(policy_score_loss), (bank,), (direction,))[1]
  numerical = (
      jax.grad(policy_score_loss)(bank + 1e-3 * direction)
      - jax.grad(policy_score_loss)(bank - 1e-3 * direction)
  ) / 2e-3
  error = jnp.max(abs(hessian_direction - numerical)) / (1 + jnp.max(abs(numerical)))
  record('second derivative directional check', float(error), 2e-2)


def _check_detached_inputs(bank, action, bandwidth, record):
  """The score loss must not differentiate through actions or advantages."""
  # Check 8: these two gradient paths are intentionally stopped, so must be zero.
  query_gradient = jax.grad(lambda query: policies.kde_policy_gradient_loss(
      bank, query, jnp.array(0.7), bandwidth
  ))(action)
  advantage_gradient = jax.grad(lambda advantage: policies.kde_policy_gradient_loss(
      bank, action, advantage, bandwidth
  ))(jnp.array(0.7))
  record('policy-score query is detached', abs(float(query_gradient)), 0)
  record('policy-score advantage is detached', abs(float(advantage_gradient)), 0)


def validate_numerics(
    bandwidths=(0.03, 0.1, 0.3), grid_size: int = 4097,
) -> ValidationResult:
  """Run formula/derivative checks; ``tables['checks']`` has pass/fail per check."""
  _settings((8,), bandwidths, (0,))
  if grid_size < 257:
    raise ValueError('Use grid_size >= 257 for numerical integration.')
  grid = jnp.linspace(-1, 1, grid_size)
  banks = {
      'interior': np.linspace(-0.7, 0.7, 8, dtype=np.float32),
      'boundary': np.array([-1, -0.99, -0.95, 0.95, 0.99, 1], np.float32),
      'outside_centres': np.array([-1.1, -0.99, -0.3, 0.4, 0.99, 1.1], np.float32),
  }
  rows = []

  def check(case, bandwidth, test, error, tolerance):
    rows.append(dict(
        case=case, bandwidth=bandwidth, check=test, error=float(error),
        tolerance=tolerance, passed=bool(np.isfinite(error) and error <= tolerance),
    ))

  fig, axes = plt.subplots(1, 2, figsize=(12, 4))
  for case, bank_np in banks.items():
    for bandwidth in bandwidths:
      density = _check_density(
          bank_np, grid, bandwidth, partial(check, case, bandwidth)
      )
      if case in ('interior', 'boundary'):
        ax = axes[0 if case == 'interior' else 1]
        ax.plot(grid, density, label=f'h={bandwidth:g}')

  bank = jnp.array([-0.9, -0.3, 0.1, 0.7, 0.99], dtype=jnp.float32)
  action = jnp.array(0.42, dtype=jnp.float32)
  for bandwidth in bandwidths:
    record = partial(check, 'derivatives', bandwidth)
    policy_score_loss = _check_first_derivatives(bank, action, bandwidth, record)
    _check_second_derivative(bank, policy_score_loss, record)
    _check_detached_inputs(bank, action, bandwidth, record)

    # Check 9: compilation must preserve the result of ordinary eager execution.
    eager = policies.kde_log_prob(bank, action, bandwidth)
    compiled = jax.jit(
        lambda b, a: policies.kde_log_prob(b, a, bandwidth)
    )(bank, action)
    check('execution', bandwidth, 'JIT versus eager', abs(float(eager - compiled)), 2e-5)
  for ax, title in zip(axes, ('Interior kernel centres', 'Boundary kernel centres')):
    ax.set(title=title, xlabel='Action', ylabel='Density')
    ax.legend()
  fig.tight_layout()
  table = _finite(pd.DataFrame(rows))
  print(f"Numerical checks passed: {int(table.passed.sum())}/{len(table)}.")
  return ValidationResult({'checks': table}, fig)


def _true_log_density(actions, mean, std):
  """Exact log density of tanh(Normal(mean, std)), including its Jacobian."""
  actions = np.asarray(actions, dtype=np.float64)
  inside = abs(actions) < 1
  safe = np.clip(actions, -1 + 1e-12, 1 - 1e-12)
  raw = np.arctanh(safe)
  logp = stats.norm.logpdf(raw, loc=mean, scale=std) - np.log1p(-safe**2)
  return np.where(inside, logp, -np.inf)


def validate_accuracy(
    sample_sizes=(32, 128, 512), bandwidths=(0.03, 0.1, 0.3),
    seeds=(601, 602, 603), grid_size: int = 2049, num_queries: int = 512,
) -> ValidationResult:
  """Measure approximation errors, not pass/fail, against known distributions.

  ``self`` metrics evaluate the KDE on its own fitting samples. ``heldout``
  metrics use fresh queries from the same distribution, avoiding self-kernel
  bias. A second, shifted distribution provides a known density-ratio target.
  """
  _settings(sample_sizes, bandwidths, seeds)
  if grid_size < 257 or num_queries < 2:
    raise ValueError('Use grid_size >= 257 and num_queries >= 2.')
  cases = {'central': (0.0, 0.5), 'narrow': (0.0, 0.1), 'near_boundary': (1.5, 0.4)}
  grid = np.linspace(-1, 1, grid_size)
  rows = []
  for case, (mean, std) in cases.items():
    # Reference: integrate the known density to obtain its true entropy.
    truth = np.exp(_true_log_density(grid, mean, std))
    true_entropy = -np.trapezoid(truth * np.log(np.maximum(truth, 1e-300)), grid)
    for n in sample_sizes:
      for seed in seeds:
        rng = np.random.default_rng(seed)
        bank = jnp.asarray(np.tanh(mean + std * rng.normal(size=n)), dtype=jnp.float32)
        queries = np.tanh(mean + std * rng.normal(size=num_queries)).astype(np.float32)
        # Use a shifted/wider policy to test target-density / source-density ratios.
        target_mean, target_std = mean + 0.2, std * 1.1
        target_bank = jnp.asarray(
            np.tanh(target_mean + target_std * rng.normal(size=n)), dtype=jnp.float32
        )
        log_ratio = (_true_log_density(queries, target_mean, target_std)
                     - _true_log_density(queries, mean, std))
        self_queries = np.asarray(bank)
        self_log_ratio = (_true_log_density(self_queries, target_mean, target_std)
                          - _true_log_density(self_queries, mean, std))
        for h in bandwidths:
          estimated = np.exp(np.asarray(_density_grid(bank, jnp.asarray(grid, jnp.float32), h)))
          logq = np.asarray(_density_grid(bank, jnp.asarray(queries), h))
          logp = np.asarray(_density_grid(target_bank, jnp.asarray(queries), h))
          self_logq = np.asarray(_density_grid(bank, bank, h))
          self_logp = np.asarray(_density_grid(target_bank, bank, h))
          rho = np.exp(logp.astype(np.float64) - logq)
          exact_rho = np.exp(log_ratio)
          rows.append(dict(
              case=case, samples=n, bandwidth=h, seed=seed,
              # Check 1: overall density error, total mass, and edge-region error.
              density_l1=np.trapezoid(abs(estimated - truth), grid),
              kde_mass=np.trapezoid(estimated, grid),
              true_mass=np.trapezoid(truth, grid),
              boundary_l1=np.trapezoid(
                  np.where(abs(grid) >= 0.8, abs(estimated - truth), 0), grid
              ),
              # Check 2: log-density accuracy on queries not used to fit the KDE.
              log_density_rmse=np.sqrt(np.mean(
                  (logq - _true_log_density(queries, mean, std))**2
              )),
              # Check 3: entropy from fitting samples versus fresh query samples.
              true_entropy=true_entropy,
              self_entropy=float(policies.kde_entropy(bank, h)),
              heldout_entropy=float(-logq.mean()),
              # Check 4: log-ratio accuracy, with and without self-kernel bias.
              ratio_log_rmse=np.sqrt(np.mean((logp - logq - log_ratio)**2)),
              self_ratio_log_rmse=np.sqrt(np.mean((self_logp - self_logq - self_log_ratio)**2)),
              # Check 5: ratio magnitudes and error after upper clipping at one.
              exact_mean_ratio=exact_rho.mean(), kde_mean_ratio=rho.mean(),
              clipped_ratio_mae=np.mean(abs(np.minimum(rho, 1) - np.minimum(exact_rho, 1))),
          ))
  table = _finite(pd.DataFrame(rows))
  table['self_entropy_error'] = table.self_entropy - table.true_entropy
  table['heldout_entropy_error'] = table.heldout_entropy - table.true_entropy
  fig, axes = plt.subplots(1, 3, figsize=(16, 4))
  for ax, case in zip(axes, cases):
    summary = table[table.case == case].groupby(['bandwidth', 'samples']).density_l1.agg(['mean', 'std'])
    for h in bandwidths:
      group = summary.loc[h]
      ax.errorbar(group.index, group['mean'], yerr=group['std'].fillna(0), marker='o', label=f'h={h:g}')
    ax.set(title=case, xlabel='Bank size', ylabel='Integrated density error', xscale='log')
    ax.legend()
  fig.tight_layout()
  print('Accuracy metrics are measurements, not automatic pass/fail thresholds.')
  print('Self-density and held-out density use separate protocols; neither is an exact likelihood.')
  return ValidationResult({'accuracy': table}, fig)


def _return_and_gradient(target):
  """Reference reward and gradient via Gaussian quadrature, not sampled actions."""
  nodes, weights = np.polynomial.hermite.hermgauss(96)
  noise = jnp.asarray(np.sqrt(2) * nodes, dtype=jnp.float32)
  weights = jnp.asarray(weights / np.sqrt(np.pi), dtype=jnp.float32)

  def expected_return(theta):
    # The two policy parameters are [Gaussian mean, log standard deviation].
    actions = jnp.tanh(theta[0] + jnp.exp(theta[1]) * noise)
    return jnp.sum(weights * -(actions - target)**2)

  return jax.jit(expected_return), jax.jit(jax.grad(expected_return))


def _score_gradient(theta, noise, independent_noise, bandwidth, method, target, baseline):
  """Estimate reward ascent using exact scores or one of two KDE bank protocols."""
  # One action per bank: column zero supplies the action whose reward is scored.
  raw = theta[0] + jnp.exp(theta[1]) * noise[:, 0]
  actions = jax.lax.stop_gradient(jnp.tanh(raw))
  raw = jax.lax.stop_gradient(raw)
  advantages = jax.lax.stop_gradient(-(actions - target)**2 - baseline)

  def loss(parameters):
    if method == 'exact':
      # The tanh Jacobian is constant with respect to parameters at fixed actions.
      logp = (-0.5 * ((raw - parameters[0]) / jnp.exp(parameters[1]))**2
              - parameters[1] - 0.5 * np.log(2 * np.pi))
      return -jnp.mean(logp * advantages)
    # Self-bank includes the scored action; independent-bank uses fresh samples.
    bank_noise = noise if method == 'kde_self_bank' else independent_noise
    banks = jnp.tanh(parameters[0] + jnp.exp(parameters[1]) * bank_noise)
    return policies.kde_policy_gradient_loss(
        banks, actions, advantages, bandwidth
    ).mean()

  return -jax.grad(loss)(theta)


def _check_gradient_estimates(
    sample_sizes, bandwidths, seeds, action_batches, methods,
    initial, expected_return, true_gradient, gradient, learning_rate,
):
  """Check 1: compare sampled ascent directions at the same initial policy."""
  reference_gradient = np.asarray(true_gradient(initial))
  baseline = expected_return(initial)
  rows = []
  for n in sample_sizes:
    for seed in seeds:
      noise, independent = jax.random.normal(
          jax.random.PRNGKey(seed), (2, action_batches, n)
      )
      for method in methods:
        for h in ((0.0,) if method == 'exact' else bandwidths):
          g = np.asarray(gradient(
              initial, noise, independent, h, method, baseline=baseline
          ))
          norm = np.linalg.norm(g)
          reference_norm = np.linalg.norm(reference_gradient)
          rows.append(dict(
              samples=n, bandwidth=h, seed=seed, estimator=method,
              gradient_mean=g[0], gradient_log_std=g[1],
              exact_gradient_mean=reference_gradient[0],
              exact_gradient_log_std=reference_gradient[1],
              # Direction agreement and magnitude error are separate measurements.
              cosine=float(g @ reference_gradient / max(norm * reference_norm, 1e-12)),
              relative_gradient_error=float(np.linalg.norm(g - reference_gradient) / reference_norm),
              gradient_norm_ratio=float(norm / reference_norm),
              # A useful direction should improve the actual reference reward.
              measured_return_change=float(
                  expected_return(initial + learning_rate * g) - baseline
              ),
          ))
  return _finite(pd.DataFrame(rows))


def _check_repeated_learning(
    seeds, methods, initial, expected_return, gradient, action_batches,
    learning_samples, learning_bandwidth, num_updates, learning_rate,
):
  """Check 2: train each estimator from scratch with fresh noise every update."""
  rows = []
  for seed in seeds:
    for method in methods:
      theta = initial
      key = jax.random.PRNGKey(seed)
      for update in range(num_updates + 1):
        value = expected_return(theta)
        rows.append(dict(
            seed=seed, estimator=method, update=update,
            action_samples=update * action_batches, expected_return=float(value),
            mean_parameter=float(theta[0]), log_std=float(theta[1]),
        ))
        if update == num_updates:
          break
        key, noise_key = jax.random.split(key)
        noise, independent = jax.random.normal(
            noise_key, (2, action_batches, learning_samples)
        )
        g = gradient(
            theta, noise, independent, learning_bandwidth, method, baseline=value
        )
        theta = theta + learning_rate * g
  return _finite(pd.DataFrame(rows))


def _plot_learning_checks(probes, learning, methods, target):
  """Show direction agreement separately from the repeated-learning curves."""
  fig, axes = plt.subplots(1, 2, figsize=(13, 4))
  for method in methods:
    selected = probes[probes.estimator == method]
    for h, group in selected.groupby('bandwidth'):
      summary = group.groupby('samples').cosine.mean()
      label = method if method == 'exact' else f'{method}, h={h:g}'
      axes[0].plot(summary.index, summary, marker='o', label=label)
    summary = learning[learning.estimator == method].groupby('update').expected_return.agg(['mean', 'std'])
    spread = summary['std'].fillna(0)
    axes[1].plot(summary.index, summary['mean'], label=method)
    axes[1].fill_between(summary.index, summary['mean'] - spread, summary['mean'] + spread, alpha=0.15)
  axes[0].set(title='Alignment with true return gradient', xlabel='Bank size', ylabel='Cosine similarity', ylim=(-1.05, 1.05), xscale='log')
  axes[1].set(title=f'One-step bounded task: reward = -(action - {target:g})²', xlabel='Policy updates', ylabel='Expected reward (quadrature)')
  for ax in axes:
    ax.legend(fontsize=7)
  fig.tight_layout()
  return fig


def validate_learning(
    sample_sizes=(32, 128, 512), bandwidths=(0.03, 0.1, 0.3),
    seeds=(701, 702, 703), action_batches: int = 256,
    learning_samples: int = 128, learning_bandwidth: float = 0.1,
    num_updates: int = 100, learning_rate: float = 0.1, target: float = 0.6,
) -> ValidationResult:
  """Test gradient usefulness on reward = -(action - target)**2, not Catch.

  ``tables['gradients']`` compares estimates at one fixed starting policy.
  ``tables['learning']`` records independent training runs for each estimator.
  The exact-score baseline uses the known policy likelihood; the two KDE
  estimators use either the action's own bank or an independently sampled bank.
  """
  _settings(sample_sizes, bandwidths, seeds)
  _settings((learning_samples,), (learning_bandwidth,), seeds)
  if (
      action_batches < 2 or num_updates < 1
      or not np.isfinite(learning_rate) or learning_rate <= 0
      or not np.isfinite(target) or abs(target) >= 1
  ):
    raise ValueError('Use positive sizes/rate, action_batches >= 2 and target strictly inside (-1, 1).')
  expected_return, true_gradient = _return_and_gradient(target)
  initial = jnp.array([0.0, -0.5])
  methods = ('exact', 'kde_self_bank', 'kde_independent_bank')
  gradient = jax.jit(
      partial(_score_gradient, target=target), static_argnames=('method', 'bandwidth')
  )

  # Check 1: does each estimated gradient agree with the true reward gradient?
  probes = _check_gradient_estimates(
      sample_sizes, bandwidths, seeds, action_batches, methods,
      initial, expected_return, true_gradient, gradient, learning_rate,
  )

  # Check 2: do repeated updates actually improve reward, beyond one good step?
  learning = _check_repeated_learning(
      seeds, methods, initial, expected_return, gradient, action_batches,
      learning_samples, learning_bandwidth, num_updates, learning_rate,
  )
  fig = _plot_learning_checks(probes, learning, methods, target)
  print('One-step task, not Catch/PPO: exact score, actor-like self-bank KDE, and independent-bank KDE.')
  print('One action is selected from each independent bank; queries/advantages are detached.')
  print('Bandwidth 0 in the probe table denotes the exact-likelihood baseline, not a KDE.')
  return ValidationResult({'gradients': probes, 'learning': learning}, fig)
