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

"""Models for conditioning on actions."""

import chex
import haiku as hk
import jax
from jax import numpy as jnp
import numpy as np

from disco_rl import types
from disco_rl import utils


def get_action_model(name: str, *args, **kwargs):
  if name == 'lstm':
    net = LSTMModel(*args, **kwargs)
  else:
    raise ValueError(f'Invalid model network name {name}.')
  return net


class LSTMModel:
  """LSTM-based action-conditional model inspried by Muesli/MuZero."""

  def __init__(
      self,
      action_spec: types.ActionSpec,
      out_spec: types.Specs,
      head_mlp_hiddens: tuple[int, ...],
      lstm_size: int,
  ) -> None:
    self._out_spec = out_spec
    self._action_spec = action_spec
    self._head_mlp_hiddens = head_mlp_hiddens
    self._lstm_size = lstm_size

  def _model_transition_all_actions(
      self, embedding: hk.LSTMState
  ) -> chex.Array:
    """Performs a model transition pass for all actions."""
    num_actions = utils.get_num_actions_from_spec(self._action_spec)
    batch_size = embedding.cell.shape[0]

    # Enumerate all action embeddings.
    one_hot_actions = jnp.eye(num_actions).astype(
        embedding.cell.dtype
    )  # [A, A]
    batched_one_hot_actions = jnp.tile(
        one_hot_actions, [batch_size, 1]
    )  # [BA, A]

    all_actions_embed = jax.tree.map(
        lambda x: jnp.repeat(x, repeats=num_actions, axis=0), embedding
    )  # [BA, *H]

    lstm_output, _ = hk.LSTM(self._lstm_size, name='action_cond')(
        batched_one_hot_actions, all_actions_embed
    )
    return lstm_output

  def _model_head_pass(
      self,
      transition_output: chex.Array,
      num_actions: int | None = None,
      policy_noise: chex.Array | None = None,
  ) -> dict[str, chex.Array]:
    """Gets outputs from MLPs with shapes according to out spec."""
    # transition_output has shape [BA, ...]
    if num_actions is None:
      num_actions = utils.get_num_actions_from_spec(self._action_spec)
    batch_size = transition_output.shape[0] // num_actions

    model_outputs = dict()
    for key, pred_spec in self._out_spec.items():
      if key == 'aux_pi' and policy_noise is not None:
        num_samples = policy_noise.shape[1]
        hidden = transition_output.reshape(batch_size, num_actions, -1)
        hidden = jnp.broadcast_to(
            hidden[:, :, None, :],
            (batch_size, num_actions, num_samples, hidden.shape[-1]),
        )
        noise = jnp.broadcast_to(
            policy_noise[:, None, :, :],
            (batch_size, num_actions, num_samples, policy_noise.shape[-1]),
        )
        pred = hk.nets.MLP(
            self._head_mlp_hiddens + (1,), name='aux_policy_generator'
        )(jnp.concatenate([hidden, noise], axis=-1))[..., 0]
        minimum, maximum = self._action_spec.minimum, self._action_spec.maximum
        model_outputs[key] = minimum + (maximum - minimum) * (
            jnp.tanh(pred) + 1.0
        ) / 2.0
        continue
      pred = hk.nets.MLP(self._head_mlp_hiddens + (np.prod(pred_spec.shape),))(
          transition_output
      )
      model_outputs[key] = pred.reshape(
          (batch_size, num_actions, *pred_spec.shape)
      )

    return model_outputs

  def model_step(
      self,
      embedding: hk.LSTMState,
      actions: chex.Array | None = None,
      policy_noise: chex.Array | None = None,
  ) -> dict[str, chex.Array]:
    """Steps model."""
    if actions is not None:
      num_actions = actions.shape[-1]
      minimum, maximum = self._action_spec.minimum, self._action_spec.maximum
      normalized = 2.0 * (actions - minimum) / (maximum - minimum) - 1.0
      action_embed = hk.Linear(16, name='continuous_action_embed')(
          normalized.reshape(-1, 1)
      )
      states = jax.tree.map(
          lambda x: jnp.repeat(x, num_actions, axis=0), embedding
      )
      transition_output, _ = hk.LSTM(self._lstm_size, name='action_cond')(
          action_embed, states
      )
      return self._model_head_pass(
          transition_output, num_actions, policy_noise
      )
    transition_output = self._model_transition_all_actions(embedding)
    model_outputs = self._model_head_pass(transition_output)
    return model_outputs

  def root_embedding(self, state: chex.Array) -> hk.LSTMState:
    """Constructs a root node from agent's state."""
    flat_state = hk.Flatten()(state)
    cell = hk.Linear(self._lstm_size)(flat_state)
    return hk.LSTMState(hidden=jnp.tanh(cell), cell=cell)
