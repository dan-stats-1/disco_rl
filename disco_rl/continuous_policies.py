"""Differentiable, scalar-action utilities for implicit policy sample banks.

These functions estimate densities of a noise-conditioned sampler; they do not
turn the sampler into an exact kernel-mixture actor. Array inputs must be real
floating NumPy/JAX arrays (including traced arrays). Settings are finite Python
or NumPy real scalars, static under JAX transformations. Array values are not
sanitized: callers are responsible for finite samples and queries.
"""

import math
import numbers

import jax
import jax.numpy as jnp
from jax.scipy import special
import numpy as np


def _floating_array(value, name: str) -> jax.Array:
  if not isinstance(value, (jax.Array, jax.core.Tracer, np.ndarray)):
    raise TypeError(f'{name} must be a NumPy or JAX array.')
  if not jnp.issubdtype(value.dtype, jnp.floating):
    raise TypeError(f'{name} must have a real floating dtype.')
  # Evaluate density arithmetic in at least single precision.
  return jnp.asarray(value, dtype=jnp.result_type(value.dtype, jnp.float32))


def _static_real(value: float, name: str) -> float:
  if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
    raise TypeError(f'{name} must be a static real scalar.')
  value = float(value)
  if not math.isfinite(value):
    raise ValueError(f'{name} must be finite.')
  return value


def _positive_setting(value: float, name: str) -> float:
  value = _static_real(value, name)
  if value <= 0:
    raise ValueError(f'{name} must be positive.')
  return value


def _kde_settings(bandwidth, minimum, maximum):
  bandwidth = _positive_setting(bandwidth, 'bandwidth')
  minimum = _static_real(minimum, 'minimum')
  maximum = _static_real(maximum, 'maximum')
  if minimum >= maximum:
    raise ValueError('minimum must be strictly less than maximum.')
  return bandwidth, minimum, maximum


def _sample_bank(samples) -> jax.Array:
  samples = _floating_array(samples, 'samples')
  if samples.ndim < 1 or samples.shape[-1] < 2:
    raise ValueError('samples must have shape [..., N] with N >= 2.')
  return samples


def smooth_wasserstein_1(
    samples_p, samples_q, smoothing: float = 0.05
) -> jax.Array:
  """Returns a smooth empirical W1 distance, with shape ``samples_p.shape[:-1]``.

  Both banks must have exactly the same shape, with a nonempty last axis.
  Independently sorted samples are paired and averaged using
  ``sqrt(residual**2 + smoothing**2) - smoothing``. The rationalized evaluation
  avoids cancellation near zero. Sorting is differentiable piecewise; ties and
  rank crossings retain JAX's sorting derivative convention.
  """
  smoothing = _positive_setting(smoothing, 'smoothing')
  samples_p = _floating_array(samples_p, 'samples_p')
  samples_q = _floating_array(samples_q, 'samples_q')
  if samples_p.shape != samples_q.shape:
    raise ValueError('Sample banks must have exactly equal shapes.')
  if samples_p.ndim < 1 or samples_p.shape[-1] == 0:
    raise ValueError('Sample banks must have a nonempty last axis.')
  residual = jnp.sort(samples_p, axis=-1) - jnp.sort(samples_q, axis=-1)
  distance = residual * (
      residual / (jnp.hypot(residual, smoothing) + smoothing)
  )
  return jnp.mean(distance, axis=-1)


def _log_normalizer(samples, bandwidth, minimum, maximum):
  lower = (minimum - samples) / bandwidth
  upper = (maximum - samples) / bandwidth
  # Reflect right-tail intervals before subtracting CDFs, avoiding two CDFs
  # rounded to one. expm1 preserves the difference of nearby log-CDF values.
  reflect = lower > -upper
  cdf_lower = jnp.where(reflect, -upper, lower)
  cdf_upper = jnp.where(reflect, -lower, upper)
  log_lower = special.log_ndtr(cdf_lower)
  log_upper = special.log_ndtr(cdf_upper)
  return log_upper + jnp.log(-jnp.expm1(log_lower - log_upper))


def _kde_log_prob_queries(samples, queries, bandwidth, minimum, maximum):
  """Evaluates queries [..., Q] against validated sample banks [..., N]."""
  residual = (queries[..., :, None] - samples[..., None, :]) / bandwidth
  log_kernels = (
      -0.5 * jnp.square(residual)
      - math.log(bandwidth)
      - 0.5 * math.log(2.0 * math.pi)
      - _log_normalizer(samples, bandwidth, minimum, maximum)[..., None, :]
  )
  log_density = special.logsumexp(log_kernels, axis=-1) - math.log(
      samples.shape[-1]
  )
  outside = (queries < minimum) | (queries > maximum)
  return jnp.where(outside, -jnp.inf, log_density)


def kde_log_prob(
    samples,
    actions,
    bandwidth: float = 0.1,
    minimum: float = -1.0,
    maximum: float = 1.0,
) -> jax.Array:
  """Estimates scalar-action log densities using a bounded Gaussian KDE.

  ``samples`` has shape [..., N], N >= 2; ``actions`` must have exactly shape
  [...], with no broadcasting. Each equal-weight Gaussian kernel is separately
  truncated and normalized on the common closed interval [minimum, maximum].
  Centers need not lie inside the interval. Out-of-range actions return -inf.
  Gradients flow through both the bank and the query; use the policy-gradient
  helper below when queries must be held fixed.
  """
  bandwidth, minimum, maximum = _kde_settings(bandwidth, minimum, maximum)
  samples = _sample_bank(samples)
  actions = _floating_array(actions, 'actions')
  if actions.shape != samples.shape[:-1]:
    raise ValueError('actions must have exactly the sample-bank prefix shape.')
  return _kde_log_prob_queries(
      samples, actions[..., None], bandwidth, minimum, maximum
  )[..., 0]


def kde_entropy(
    samples,
    bandwidth: float = 0.1,
    minimum: float = -1.0,
    maximum: float = 1.0,
) -> jax.Array:
  """Returns the negative mean self-density log probability, shape [...].

  This is a self-density plug-in estimate, not an unbiased entropy estimator:
  evaluating on the fitting bank introduces finite-sample/self-kernel bias.
  Both density-bank and query paths remain differentiable, so this is the
  derivative of the actual plug-in objective, not a score-type surrogate.
  Samples outside the bounds consequently produce infinite estimated entropy.
  """
  bandwidth, minimum, maximum = _kde_settings(bandwidth, minimum, maximum)
  samples = _sample_bank(samples)
  return -jnp.mean(
      _kde_log_prob_queries(
          samples, samples, bandwidth, minimum, maximum
      ),
      axis=-1,
  )


def kde_policy_gradient_loss(
    samples,
    actions,
    advantages,
    bandwidth: float = 0.1,
    minimum: float = -1.0,
    maximum: float = 1.0,
    backprop: bool = False,
) -> jax.Array:
  """Returns per-action ``-log_density * advantage``, with prefix shape [...].

  Actions are always held fixed for score-function differentiation. Gradients
  to the density samples are preserved, including higher-order derivatives.
  Advantages must have exactly the action shape and are stop-gradient unless
  ``backprop=True``. Settings, including backprop, must be static under jit.
  This uses an estimated sampler density, not an exact sampler likelihood.
  """
  if not isinstance(backprop, bool):
    raise TypeError('backprop must be a static bool.')
  actions = _floating_array(actions, 'actions')
  advantages = _floating_array(advantages, 'advantages')
  if advantages.shape != actions.shape:
    raise ValueError('advantages must have exactly the action shape.')
  log_probability = kde_log_prob(
      samples,
      jax.lax.stop_gradient(actions),
      bandwidth,
      minimum,
      maximum,
  )
  if not backprop:
    advantages = jax.lax.stop_gradient(advantages)
  return -log_probability * advantages
