# Copyright 2019, Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""An implementation of the FedAvg algorithm with learning rate schedules.

This is intended to be a somewhat minimal implementation of Federated
Averaging that allows for client and server learning rate scheduling.

The original FedAvg is based on the paper:

Communication-Efficient Learning of Deep Networks from Decentralized Data
    H. Brendan McMahan, Eider Moore, Daniel Ramage,
    Seth Hampson, Blaise Aguera y Arcas. AISTATS 2017.
    https://arxiv.org/abs/1602.05629
"""

import attr
import collections
from typing import Any, Union, Callable, Optional

import numpy as np
import tensorflow as tf

import tensorflow_federated as tff
from tensorflow_federated.python.learning import models
from tensorflow_federated.python.tensorflow_libs import tensor_utils
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.impl.types import type_analysis
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import type_conversions
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.core.impl.computation import computation_base
from tensorflow_federated.python.core.impl.federated_context import intrinsics

# Convenience type aliases.
ModelBuilder = Callable[[], Union[tff.learning.models.VariableModel, tff.learning.models.FunctionalModel, tff.learning.models.ReconstructionModel]]
OptimizerBuilder = Callable[[float], tf.keras.optimizers.Optimizer]
ClientWeightFn = Callable[..., float]
LRScheduleFn = Callable[[Union[int, tf.Tensor]], Union[tf.Tensor, float]]


def _initialize_optimizer_vars(model: Union[tff.learning.models.VariableModel, tff.learning.models.FunctionalModel, tff.learning.models.ReconstructionModel],
                               optimizer: tf.keras.optimizers.Optimizer):
  """Ensures variables holding the state of `optimizer` are created."""
  delta = tf.nest.map_structure(tf.zeros_like, _get_weights(model).trainable)
  model_weights = _get_weights(model)
  grads_and_vars = tf.nest.map_structure(lambda x, v: (x, v), delta,
                                         model_weights.trainable)
  optimizer.apply_gradients(grads_and_vars, name='server_update')
  assert optimizer.variables()


def _get_weights(model: Union[tff.learning.models.VariableModel, tff.learning.models.FunctionalModel, tff.learning.models.ReconstructionModel]) -> tff.learning.models.ModelWeights:
  return tff.learning.models.ModelWeights.from_model(model)


@attr.s(eq=False, order=False, frozen=True)
class ServerState(object):
  """Structure for state on the server.

  Fields:
  -   `model`: A dictionary of the model's trainable and non-trainable
        weights.
  -   `optimizer_state`: The server optimizer variables.
  -   `round_num`: The current training round, as a float.
  -    sampling_rates: tensor storing client's sampling rates. 
  """
  model = attr.ib()
  optimizer_state = attr.ib()
  round_num = attr.ib()
  aggregation_state = attr.ib()
  # This is a float to avoid type incompatibility when calculating learning rate
  # schedules.


@tf.function
def server_update(model, server_optimizer, server_state, weights_delta):
  """Updates `server_state` based on `weights_delta`, increase the round number.

  Args:
    model: A `Union[tff.learning.models.VariableModel, tff.learning.models.FunctionalModel, tff.learning.models.ReconstructionModel]`.
    server_optimizer: A `tf.keras.optimizers.Optimizer`.
    server_state: A `ServerState`, the state to be updated.
    weights_delta: An update to the trainable variables of the model.

  Returns:
    An updated `ServerState`.
  """
  model_weights = _get_weights(model)
  tff.learning.models.ModelWeights(model_weights, server_state.model)
  # Server optimizer variables must be initialized prior to invoking this
  tff.learning.models.ModelWeights(server_optimizer.variables(), server_state.optimizer_state)

  weights_delta, has_non_finite_weight = (
      tensor_utils.zero_all_if_any_non_finite(weights_delta))
  if has_non_finite_weight > 0:
    return server_state

  # Apply the update to the model. We must multiply weights_delta by -1.0 to
  # view it as a gradient that should be applied to the server_optimizer.
  grads_and_vars = [
      (-1.0 * x, v) for x, v in zip(weights_delta, model_weights.trainable)
  ]

  server_optimizer.apply_gradients(grads_and_vars)

  # Create a new state based on the updated model.
  return tff.structure.update_struct(
      server_state,
      model=model_weights,
      optimizer_state=server_optimizer.variables(),
      round_num=server_state.round_num + 1.0)


@attr.s(eq=False, order=False, frozen=True)
class ClientOutput(object):
  """Structure for outputs returned from clients during federated optimization.

  Fields:
  -   `weights_delta`: A dictionary of updates to the model's trainable
        variables.
  -   `client_weight`: Weight to be used in a weighted mean when
        aggregating `weights_delta`.
  -   `model_output`: A structure matching `Union[tff.learning.models.VariableModel, 
        tff.learning.models.FunctionalModel, tff.learning.models.ReconstructionModel].report_local_unfinalized_metrics`, 
        reflecting the results of training on the input dataset.
  -   `optimizer_output`: Additional metrics or other outputs defined by the
        optimizer.
  """
  weights_delta = attr.ib()
  client_weight = attr.ib()
  model_output = attr.ib()
  optimizer_output = attr.ib()


def create_client_update_fn():
  """Returns a tf.function for the client_update.

  This "create" fn is necesessary to prevent
  "ValueError: Creating variables on a non-first call to a function decorated
  with tf.function" errors due to the client optimizer creating variables. This
  is really only needed because we test the client_update function directly.
  """
  @tf.function
  def client_update(model,
                    dataset,
                    initial_weights,
                    client_optimizer):
    """Updates client model.

    Args:
      model: A `Union[tff.learning.models.VariableModel, tff.learning.models.FunctionalModel, tff.learning.models.ReconstructionModel]`.
      dataset: A 'tf.data.Dataset'.
      initial_weights: A `tff.learning.models.ModelWeights` from server.
      client_optimizer: A `tf.keras.optimizer.Optimizer` object.

    Returns:
      A 'ClientOutput`.
    """

    model_weights = _get_weights(model)
    tff.learning.models.ModelWeights(model_weights, initial_weights)

    num_examples = tf.constant(0, dtype=tf.int32)
    for batch in dataset:
      with tf.GradientTape() as tape:
        output = model.forward_pass(batch)
      grads = tape.gradient(output.loss, model_weights.trainable)
      grads_and_vars = zip(grads, model_weights.trainable)
      client_optimizer.apply_gradients(grads_and_vars)
      num_examples += tf.shape(output.predictions)[0]

    model_output = aggregated_outputs = model.report_local_unfinalized_metrics()

    weights_delta = tf.nest.map_structure(lambda a, b: a - b,
                                          model_weights.trainable,
                                          initial_weights.trainable)
    weights_delta, has_non_finite_weight = (
        tensor_utils.zero_all_if_any_non_finite(weights_delta))

    if has_non_finite_weight > 0:
      client_weight = tf.constant(0, dtype=tf.float32)
    else:
      client_weight = tf.constant(1, dtype=tf.float32)
    #elif client_weight_fn is None:
      #client_weight = tf.cast(num_examples, dtype=tf.float32)
    #else:
      #client_weight = client_weight_fn(aggregated_outputs)
      
    optimizer_output = collections.OrderedDict([('num_examples', num_examples)])

    return ClientOutput(weights_delta, client_weight, model_output, optimizer_output)

  return client_update


def build_server_init_fn(
    *,
    model_fn: ModelBuilder,
    server_optimizer_fn: Callable[[], tf.keras.optimizers.Optimizer],
    aggregation_process: Optional[tff.templates.MeasuredProcess])-> computation_base.Computation:
  """Builds a `tff.tf_computation` that returns the initial `ServerState`.

  The attributes `ServerState.model` and `ServerState.optimizer_state` are
  initialized via their constructor functions. The attribute
  `ServerState.round_num` is set to 0.0.

  Args:
    model_fn: A no-arg function that returns a `Union[tff.learning.models.VariableModel, tff.learning.models.FunctionalModel, tff.learning.models.ReconstructionModel]`.
    server_optimizer_fn: A no-arg function that returns a
      `tf.keras.optimizers.Optimizer`.

  Returns:
    A `tff.tf_computation` that returns initial `ServerState`.
  """

  @tff.tf_computation
  def server_init_tf():
    server_optimizer = server_optimizer_fn()
    model = model_fn()
    _initialize_optimizer_vars(model, server_optimizer)
    return _get_weights(model), server_optimizer.variables()
    # return ServerState(
    #     model=_get_weights(model),
    #     optimizer_state=server_optimizer.variables(),
    #     round_num=0.0, 
    #     delta_aggregate_state=aggregation_process.initialize())

  @tff.federated_computation()
  def initialize_computation():
    model = model_fn()
    initial_global_model, initial_global_optimizer_state = intrinsics.federated_eval(
        server_init_tf, tff.SERVER)
    return intrinsics.federated_zip(ServerState(
        model=initial_global_model,
        optimizer_state=initial_global_optimizer_state,
        round_num=tff.federated_value(0.0, tff.SERVER),
        aggregation_state=aggregation_process.initialize(),
        ))

  return initialize_computation


def build_fed_avg_process(
    model_fn: ModelBuilder,
    client_optimizer_fn: OptimizerBuilder,
    client_lr: Union[float, LRScheduleFn] = 0.1,
    server_optimizer_fn: OptimizerBuilder = tf.keras.optimizers.SGD,
    server_lr: Union[float, LRScheduleFn] = 1.0,
    aggregation_process: Optional[tff.templates.MeasuredProcess] = None,
) -> tff.templates.IterativeProcess:
  """Builds the TFF computations for optimization using federated averaging.

  Args:
    model_fn: A no-arg function that returns a `Union[tff.learning.models.VariableModel, tff.learning.models.FunctionalModel, tff.learning.models.ReconstructionModel]`.
    client_optimizer_fn: A function that accepts a `learning_rate` keyword
      argument and returns a `tf.keras.optimizers.Optimizer` instance.
    client_lr: A scalar learning rate or a function that accepts a float
      `round_num` argument and returns a learning rate.
    server_optimizer_fn: A function that accepts a `learning_rate` argument and
      returns a `tf.keras.optimizers.Optimizer` instance.
    server_lr: A scalar learning rate or a function that accepts a float
      `round_num` argument and returns a learning rate.
    client_weight_fn: Optional function that takes the output of
      `model.report_local_outputs` and returns a tensor that provides the weight
      in the federated average of model deltas. If not provided, the default is
      the total number of examples processed on device.

  Returns:
    A `tff.templates.IterativeProcess`.
  """

  client_lr_schedule = client_lr
  if not callable(client_lr_schedule):
    client_lr_schedule = lambda round_num: client_lr

  server_lr_schedule = server_lr
  if not callable(server_lr_schedule):
    server_lr_schedule = lambda round_num: server_lr


  with tf.Graph().as_default():
    dummy_model = model_fn()
    model_weights_type = models.weights_type_from_model(
        dummy_model)
    dummy_optimizer = server_optimizer_fn()
    _initialize_optimizer_vars(dummy_model, dummy_optimizer)
    optimizer_variable_type = tff.types.type_from_tensors(
        dummy_optimizer.variables())    

  initialize_computation = build_server_init_fn(
      model_fn = model_fn,
      # Initialize with the learning rate for round zero.
      server_optimizer_fn = lambda: server_optimizer_fn(server_lr_schedule(0)),
      aggregation_process = aggregation_process)
  
  #model_weights_type = tff.types.type_from_tensors(_get_weights(dummy_model).trainable)
  round_num_type = tf.float32

  tf_dataset_type = tff.SequenceType(dummy_model.input_spec)
  model_input_type = tff.SequenceType(dummy_model.input_spec)
  client_weight_type = tf.float32

  aggregation_state_type = aggregation_process.initialize.type_signature.result.member

  server_state_type = ServerState(
      model=model_weights_type,
      optimizer_state=optimizer_variable_type,
      round_num=round_num_type,
      aggregation_state=aggregation_state_type,
      )


  @tff.tf_computation(model_input_type, model_weights_type, round_num_type)
  def client_update_fn(tf_dataset, initial_model_weights, round_num):
    client_lr = client_lr_schedule(round_num)
    client_optimizer = client_optimizer_fn(client_lr)
    client_update = create_client_update_fn()
    return client_update(model_fn(), tf_dataset, initial_model_weights,
                         client_optimizer)


  @tff.tf_computation(server_state_type, model_weights_type.trainable)
  def server_update_fn(server_state, model_delta):
    model = model_fn()
    server_lr = server_lr_schedule(server_state.round_num)
    server_optimizer = server_optimizer_fn(server_lr)
    # We initialize the server optimizer variables to avoid creating them
    # within the scope of the tf.function server_update.
    _initialize_optimizer_vars(model, server_optimizer)
    return server_update(model, server_optimizer, server_state, model_delta)


  # @tff.tf_computation(tf.float32, tf.float32)
  # def local_mul(weight, participated):
  #   return tf.math.multiply(weight, participated)

  @tff.federated_computation(
      tff.FederatedType(server_state_type, tff.SERVER),
      tff.FederatedType(tf_dataset_type, tff.CLIENTS),
      tff.FederatedType(client_weight_type, tff.CLIENTS))
  def run_one_round(server_state, federated_dataset, client_weight):
    """Orchestration logic for one round of computation.

    Args:
      server_state: A `ServerState`.
      federated_dataset: A federated `tf.Dataset` with placement `tff.CLIENTS`.

    Returns:
      A tuple of updated `ServerState` and the result of
      `Union[tff.learning.models.VariableModel, tff.learning.models.FunctionalModel, tff.learning.models.ReconstructionModel].report_local_unfinalized_metrics`.
    """
    client_model = tff.federated_broadcast(server_state.model)
    client_round_num = tff.federated_broadcast(server_state.round_num)

    client_outputs = tff.federated_map(
        client_update_fn,
        (federated_dataset, client_model,client_round_num))

    #client_weight = client_outputs.client_weight
    # model_delta = tff.federated_mean(
    #     client_outputs.weights_delta, weight=client_weight)

    participant_client_weight = tff.federated_map(
      tff.tf_computation(lambda x,y: x*y), 
      (client_weight,client_outputs.client_weight))

    aggregation_output = aggregation_process.next(
        server_state.aggregation_state, client_outputs.weights_delta,
        participant_client_weight)   

    server_state = tff.federated_map(server_update_fn,
                                     (server_state, aggregation_output.result))
    # V.18
    # aggregated_outputs = dummy_model.federated_output_computation(
    #     client_outputs.model_output)
    # if aggregated_outputs.type_signature.is_struct():
    #   aggregated_outputs = tff.federated_zip(aggregated_outputs)

    aggregated_outputs = client_outputs.model_output

    return server_state, aggregated_outputs

  # @tff.federated_computation
  # def initialize_fn():
  #   return tff.federated_value(server_init_tf(), tff.SERVER)

  return tff.templates.IterativeProcess(
      initialize_fn=initialize_computation, next_fn=run_one_round)
