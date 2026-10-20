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

"""Network factory."""

from collections.abc import Iterable, Mapping
from typing import Any

import chex
import haiku as hk
from haiku import initializers as hk_init
import jax
from jax import numpy as jnp
import numpy as np

from disco_rl import types
from disco_rl.networks import action_models


def get_network(name: str, *args, **kwargs) -> types.PolicyNetwork:
  """Constructs a network."""

  def _get_net():
    if name == 'mlp':
      return MLP(*args, **kwargs)
    else:
      raise ValueError(f'Unknown network: {name}')

  def _agent_step(*call_args, **call_kwargs):
    return _get_net()(*call_args, **call_kwargs)

  def _unroll(*call_args, **call_kwargs):
    return _get_net().unroll(*call_args, **call_kwargs)

  module_init_fn, one_step_fn = hk.without_apply_rng(
      hk.transform_with_state(_agent_step)
  )
  _, unroll_fn = hk.without_apply_rng(hk.transform_with_state(_unroll))

  return types.PolicyNetwork(
      init=module_init_fn,
      one_step=one_step_fn,
      unroll=unroll_fn,
  )


class MLPHeadNet(hk.Module):
  """MLP heads according to the update rules' out_spec."""

  def __init__(
      self,
      out_spec: chex.ArrayTree,
      action_spec: types.Specs,
      head_w_init_std: float | None,
      model_out_spec: chex.ArrayTree | None = None,
      model_arch_name: str | None = None,
      model_kwargs: Mapping[str, Any] | None = None,
      module_name: str | None = None,
  ):
    super().__init__(name=module_name)
    self._out_spec = out_spec
    self._action_spec = action_spec
    if model_out_spec:
      self._model = action_models.get_action_model(
          model_arch_name,
          action_spec=action_spec,
          out_spec=model_out_spec,
          **model_kwargs,
      )
    else:
      self._model = None
    self._head_w_init = (
        hk_init.TruncatedNormal(head_w_init_std) if head_w_init_std else None
    )

  def _embedding_pass(
      self, inputs: chex.ArrayTree, should_reset: chex.Array | None = None
  ) -> chex.Array:
    """Compute embedding from agent inputs."""
    raise NotImplementedError

  def _head_pass(
      self, embedding: chex.Array, policy_noise: chex.Array | None = None
  ) -> dict[str, chex.Array]:
    """Compute outputs as linear functions of embedding."""
    embedding = hk.Flatten()(embedding)

    def _infer(spec: types.ArraySpec) -> chex.Array:
      output = hk.nets.MLP(
          output_sizes=(np.prod(spec.shape),),
          w_init=self._head_w_init,
          name='torso_head',
      )(embedding)
      output = output.reshape((embedding.shape[0], *spec.shape))
      return output

    if policy_noise is None:
      return jax.tree.map(_infer, self._out_spec)
    chex.assert_shape(
        policy_noise,
        (embedding.shape[0], self._out_spec['logits'].shape[0], 1),
    )
    out = {
        key: _infer(spec)
        for key, spec in self._out_spec.items()
        if key != 'logits'
    }
    sample_shape = policy_noise.shape[:2]
    state = jnp.broadcast_to(
        embedding[:, None, :], (*sample_shape, embedding.shape[-1])
    )
    generator_input = jnp.concatenate([state, policy_noise], axis=-1)
    samples = hk.nets.MLP((64, 64, 1), name='policy_generator')(generator_input)
    minimum, maximum = self._action_spec.minimum, self._action_spec.maximum
    out['logits'] = minimum + (maximum - minimum) * (
        jnp.tanh(samples[..., 0]) + 1.0
    ) / 2.0
    return out

  def unroll(
      self, inputs: chex.ArrayTree, should_reset: chex.Array | None = None
  ) -> dict[str, chex.Array]:
    """Assumes there is a time dimension in the inputs."""
    return hk.BatchApply(self.__call__)(inputs, should_reset)

  def __call__(
      self, inputs: chex.ArrayTree, should_reset: chex.Array | None = None
  ) -> dict[str, chex.Array]:
    torso = self._embedding_pass(inputs)
    continuous = (
        'logits' in self._out_spec
        and hasattr(self._action_spec, 'dtype')
        and np.issubdtype(self._action_spec.dtype, np.floating)
    )
    policy_noise = None
    if continuous:
      if not isinstance(inputs, dict) or '_policy_noise' not in inputs:
        raise ValueError('Continuous policy inputs require _policy_noise.')
      policy_noise = inputs['_policy_noise']
    out = self._head_pass(torso, policy_noise)
    if self._model:
      root = self._model.root_embedding(torso)
      if continuous:
        actions = jnp.concatenate(
            [
                inputs.get('_model_samples', out['logits']),
                inputs['_model_action'][..., None],
            ],
            axis=-1,
        )
        model_out = self._model.model_step(root, actions, policy_noise)
        out.update({key + '_a': value[:, -1] for key, value in model_out.items()})
        model_out = {key: value[:, :-1] for key, value in model_out.items()}
      else:
        model_out = self._model.model_step(root)
      out.update(model_out)
    return out


class MLP(MLPHeadNet):
  """Simple MLP network."""

  def __init__(
      self,
      out_spec: chex.ArrayTree,
      dense: Iterable[int],
      action_spec: types.Specs,
      head_w_init_std: float | None,
      model_out_spec: chex.ArrayTree | None = None,
      model_arch_name: str | None = None,
      model_kwargs: Mapping[str, Any] | None = None,
      module_name: str | None = None,
  ) -> None:
    super().__init__(
        out_spec,
        action_spec=action_spec,
        model_out_spec=model_out_spec,
        head_w_init_std=head_w_init_std,
        model_arch_name=model_arch_name,
        model_kwargs=model_kwargs,
        module_name=module_name,
    )
    self._dense = dense

  def _embedding_pass(
      self, inputs: chex.ArrayTree, should_reset: chex.Array | None = None
  ) -> chex.Array:
    del should_reset
    if isinstance(inputs, dict):
      inputs = {
          key: value
          for key, value in inputs.items()
          if key not in ('_policy_noise', '_model_action', '_model_samples')
      }
    inputs = [hk.Flatten()(x) for x in jax.tree_util.tree_leaves(inputs)]
    inputs = jnp.concatenate(inputs, axis=-1)
    return hk.nets.MLP(self._dense, name='torso')(inputs)
