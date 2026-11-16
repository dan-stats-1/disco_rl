"""Shared actor collection for jittable environments."""

import jax

from disco_rl import types


def unroll_jittable_actor(
    params, actor_state, ts, env_state, rng, env, rollout_len, actor_step_fn,
):
  """Collect actor timesteps while retaining policy noise and starting states."""
  def single_step(carry, step_rng):
    env_state, ts, actor_state = carry
    actor_timestep, actor_state = actor_step_fn(
        params, step_rng, ts, actor_state
    )
    env_state, ts = env.step(env_state, actor_timestep.actions)
    return (env_state, ts, actor_state), actor_timestep

  (env_state, ts, actor_state), actor_rollout = jax.lax.scan(
      single_step, (env_state, ts, actor_state),
      jax.random.split(rng, rollout_len),
  )
  return (
      types.ActorRollout.from_timestep(actor_rollout),
      actor_state, ts, env_state,
  )
