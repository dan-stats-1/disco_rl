# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Jittable environments."""

import math

from dm_env import specs
import jax
import jax.numpy as jnp
from ml_collections import config_dict

from disco_rl.environments.wrappers import batched_jittable_env


class _SingleStreamCatch:
  """Catch with lifetime reset."""

  def __init__(self, rows: int = 10, columns: int = 5):
    self._rows = rows
    self._columns = columns

  @property
  def num_actions(self) -> int:
    return 3

  def initial_state(self, rng):
    ball_y = 0
    ball_x = jax.random.randint(rng, (), 0, self._columns)
    paddle_y = self._rows - 1
    paddle_x = self._columns // 2
    return jnp.array((ball_y, ball_x, paddle_y, paddle_x), dtype=jnp.int32)

  def episode_reset(self, rng, state):
    del state
    return self.initial_state(rng)

  def step(self, rng, state, action):
    del rng
    paddle_x = jnp.clip(state[3] + action - 1, 0, self._columns - 1)
    return jnp.array([state[0] + 1, state[1], state[2], paddle_x])

  def is_terminal(self, state):
    return state[0] == self._rows - 1

  def render(self, state):
    """Render the screen."""

    def f(y, x):
      return jax.lax.select(
          jnp.bitwise_or(
              jnp.bitwise_and(y == state[0], x == state[1]),
              jnp.bitwise_and(y == state[2], x == state[3]),
          ),
          1.0,
          0.0,
      )

    y_board = jnp.repeat(jnp.arange(self._rows), self._columns)
    x_board = jnp.tile(jnp.arange(self._columns), self._rows)
    return jax.vmap(f)(y_board, x_board).reshape((self._rows, self._columns, 1))

  def reward(self, state):
    return jax.lax.select(
        state[0] == self._rows - 1,
        jax.lax.select(state[1] == state[3], 1.0, -1.0),
        0.0,
    )


class CatchJittableEnvironment(batched_jittable_env.BatchedJittableEnvironment):
  """Catch environment."""

  def __init__(
      self,
      batch_size: int,
      env_settings: config_dict.ConfigDict,
  ) -> None:

    super().__init__(
        _SingleStreamCatch,
        batch_size,
        env_settings,
    )


def get_config_catch() -> config_dict.ConfigDict:
  """Returns default config for CatchEnvironment."""
  return config_dict.ConfigDict(
      dict(
          rows=8,
          columns=8,
      )
  )


class _SingleStreamContinuousCatch:
  """Catch with a continuous paddle position and direct velocity control.

  State is [ball_y, ball_x, paddle_y, paddle_x], stored as float32. The
  ball falls one row per step. Actions in [-1, 1] move the paddle by
  action * max_speed columns without inertia. Out-of-range actions are clipped.
  """

  def __init__(
      self,
      rows: int = 8,
      columns: int = 8,
      max_speed: float = 1.0,
      catch_radius: float = 0.5,
  ):
    if not isinstance(rows, int) or rows < 2:
      raise ValueError('rows must be an integer greater than or equal to 2.')
    if not isinstance(columns, int) or columns < 2:
      raise ValueError('columns must be an integer greater than or equal to 2.')
    if not math.isfinite(max_speed) or max_speed <= 0:
      raise ValueError('max_speed must be finite and positive.')
    if not math.isfinite(catch_radius) or catch_radius <= 0:
      raise ValueError('catch_radius must be finite and positive.')
    self._rows = rows
    self._columns = columns
    self._max_speed = max_speed
    self._catch_radius = catch_radius

  def initial_state(self, rng):
    ball_x = jax.random.uniform(
        rng, (), dtype=jnp.float32, minval=0.0, maxval=self._columns - 1.0
    )
    return jnp.array(
        [0.0, ball_x, self._rows - 1.0, (self._columns - 1.0) / 2.0],
        dtype=jnp.float32,
    )

  def episode_reset(self, rng, state):
    del state
    return self.initial_state(rng)

  def step(self, rng, state, action):
    del rng
    velocity = jnp.clip(action, -1.0, 1.0) * self._max_speed
    paddle_x = jnp.clip(state[3] + velocity, 0.0, self._columns - 1.0)
    return state.at[0].add(1.0).at[3].set(paddle_x)

  def is_terminal(self, state):
    return state[0] >= state[2]

  def render(self, state):
    """Returns normalized [ball_y, ball_x, paddle_x], without rasterization."""
    return jnp.array(
        [
            state[0] / (self._rows - 1.0),
            state[1] / (self._columns - 1.0),
            state[3] / (self._columns - 1.0),
        ],
        dtype=jnp.float32,
    )

  def reward(self, state):
    caught = jnp.abs(state[1] - state[3]) <= self._catch_radius
    return jnp.where(self.is_terminal(state), jnp.where(caught, 1.0, -1.0), 0.0)


class ContinuousCatchJittableEnvironment(
    batched_jittable_env.BatchedJittableEnvironment
):
  """Batched continuous Catch with scalar float32 actions in [-1, 1]."""

  def __init__(
      self,
      batch_size: int,
      env_settings: config_dict.ConfigDict,
  ) -> None:
    super().__init__(
        _SingleStreamContinuousCatch,
        batch_size,
        env_settings,
        action_spec=specs.BoundedArray((), jnp.float32, -1.0, 1.0),
    )


def get_config_continuous_catch() -> config_dict.ConfigDict:
  """Returns the default configuration for continuous Catch."""
  return config_dict.ConfigDict(
      dict(
          rows=8,
          columns=8,
          max_speed=1.0,
          catch_radius=0.5,
      )
  )
