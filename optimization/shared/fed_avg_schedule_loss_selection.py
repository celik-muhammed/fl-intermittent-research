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
from absl import logging

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

import tensorflow_federated as tff
from tensorflow.python.ops import clip_ops
from tensorflow_model_optimization.python.core.internal import tensor_encoding as te
from tensorflow_federated.python.learning import models
from tensorflow_federated.python.tensorflow_libs import tensor_utils
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.impl.types import type_analysis
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import type_conversions
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.core.impl.computation import computation_base
from tensorflow_federated.python.core.impl.federated_context import intrinsics
from tensorflow_federated.python.core.templates import measured_process

NONE_SERVER_TYPE = computation_types.FederatedType((), placements.SERVER)


# Convenience type aliases.
ModelBuilder = Callable[[], Union[tff.learning.models.VariableModel, tff.learning.models.FunctionalModel, tff.learning.models.ReconstructionModel]]
OptimizerBuilder = Callable[[float], tf.keras.optimizers.Optimizer]
ClientWeightFn = Callable[..., float]
LRScheduleFn = Callable[[Union[int, tf.Tensor]], Union[tf.Tensor, float]]

@tff.federated_computation()
def _empty_server_initialization():
  return intrinsics.federated_value([], placements.SERVER)

def _is_valid_stateful_process(
    process: measured_process.MeasuredProcess) -> bool:
  """Validates whether a `MeasuredProcess` is valid for model delta processes.
  Valid processes must have `state` and `measurements` placed on the server.
  This method is intended to be used with additional validation on the non-state
  parameters, inputs and result.
  Args:
    process: A measured process to validate.
  Returns:
    `True` iff `process` is a valid stateful process, `False` otherwise.
  """
  init_type = process.initialize.type_signature
  next_type = process.next.type_signature
  return (init_type.result.placement is placements.SERVER and
          next_type.parameter[0].placement is placements.SERVER and
          next_type.result.state.placement is placements.SERVER and
          next_type.result.measurements.placement is placements.SERVER)

def _is_valid_aggregation_process(
    process: measured_process.MeasuredProcess) -> bool:
  """Validates a `MeasuredProcess` adheres to the aggregation signature.
  A valid aggregation process is one whose argument is placed at `SERVER` and
  whose output is placed at `CLIENTS`.
  Args:
    process: A measured process to validate.
  Returns:
    `True` iff the process is a validate aggregation process, otherwise `False`.
  """
  next_type = process.next.type_signature
  return (isinstance(process, measured_process.MeasuredProcess) and
          _is_valid_stateful_process(process) and
          next_type.parameter[1].placement is placements.CLIENTS and
          next_type.result.result.placement is placements.SERVER)

def build_stateless_mean(
    *, model_delta_type: Union[computation_types.StructType,
                               computation_types.TensorType]
) -> measured_process.MeasuredProcess:
  """Builds a `MeasuredProcess` that wraps` tff.federated_mean`."""

  @tff.federated_computation(
      NONE_SERVER_TYPE,
      computation_types.FederatedType(model_delta_type, placements.CLIENTS),
      computation_types.FederatedType(tf.float32, placements.CLIENTS))
  def stateless_mean(state, value, weight):
    empty_metrics = intrinsics.federated_value((), placements.SERVER)
    return measured_process.MeasuredProcessOutput(
        state=state,
        result=intrinsics.federated_mean(value, weight=weight),
        measurements=empty_metrics)

  return measured_process.MeasuredProcess(
      initialize_fn=_empty_server_initialization, next_fn=stateless_mean)


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


@attr.s(eq=False, frozen=True)
class ServerState(object):
  """Structure for state on the server.

  Fields:
  -   `model`: A dictionary of the model's trainable and non-trainable
        weights.
  -   `optimizer_state`: The server optimizer variables.
  -   `round_num`: The current training round, as a float.
  """
  model = attr.ib()
  optimizer_state = attr.ib()
  round_num = attr.ib()
  effective_num_clients = attr.ib()
  delta_aggregate_state = attr.ib()
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
  client_id = attr.ib()

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
                    client_optimizer,
                    client_id,
                    client_weight_fn=None):
    """Updates client model.

    Args:
      model: A `Union[tff.learning.models.VariableModel, tff.learning.models.FunctionalModel, tff.learning.models.ReconstructionModel]`.
      dataset: A 'tf.data.Dataset'.
      initial_weights: A `tff.learning.models.ModelWeights` from server.
      client_optimizer: A `tf.keras.optimizer.Optimizer` object.
      client_weight_fn: Optional function that takes the output of `Union[tff.learning.models.VariableModel, 
        tff.learning.models.FunctionalModel, tff.learning.models.ReconstructionModel].report_local_unfinalized_metrics`, 
        reflecting the results of training on the input dataset. and returns a tensor that provides the
        weight in the federated average of model deltas. If not provided, the
        default is the total number of examples processed on device.

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
      #grads = tf.nest.map_structure(lambda g: clip_ops.clip_by_norm(g,5.0), grads)
      grads, _ = tf.clip_by_global_norm(grads, 1.0)
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
      client_weight = tf.constant([[0]], dtype=tf.float32)
    else :
      client_weight = tf.cast([[num_examples]], dtype=tf.float32)
    # else:
      # client_weight = client_weight_fn(aggregated_outputs)

    #weights_delta_encoded = tf.nest.map_structure(mean_encoder_fn, weights_delta)
      
    optimizer_output = collections.OrderedDict([('num_examples', num_examples)])

    return ClientOutput(weights_delta, client_weight,  model_output, optimizer_output, client_id)

  return client_update


def build_server_init_fn(
  *,
  model_fn: ModelBuilder,
  effective_num_clients: int,
  server_optimizer_fn: Callable[[], tf.keras.optimizers.Optimizer],
  aggregation_process: Optional[measured_process.MeasuredProcess])-> computation_base.Computation:
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

  @tff.tf_computation()
  def server_init_tf():
    server_optimizer = server_optimizer_fn()
    model = model_fn()
    _initialize_optimizer_vars(model, server_optimizer)
    return _get_weights(model), server_optimizer.variables(),


  @tff.tf_computation()
  def get_effective_num_clients():
    return tf.constant(effective_num_clients, dtype=tf.int32)
   

  @tff.federated_computation()
  def initialize_computation():
    model = model_fn()
    initial_global_model, initial_global_optimizer_state = intrinsics.federated_eval(
        server_init_tf, placements.SERVER)
    return intrinsics.federated_zip(ServerState(
        model=initial_global_model,
        optimizer_state=initial_global_optimizer_state,
        round_num=tff.federated_value(0.0, tff.SERVER),
        effective_num_clients= intrinsics.federated_eval(get_effective_num_clients, placements.SERVER),
        delta_aggregate_state=aggregation_process.initialize(),
        ))

  return initialize_computation

@tf.function
def redefine_client_weight( losses,weights, effective_num_clients):
  flat_weights = tf.reshape(weights, shape = [-1])
  flat_loss = tf.reshape(tf.convert_to_tensor(losses, dtype = tf.float32), shape = [-1])
  new_weights = tf.zeros_like(weights, tf.float32)
  values, indices = tf.math.top_k(flat_loss, k=effective_num_clients, sorted=False)
  expanded_indices = tf.expand_dims(indices, axis=1)
  keep_weights = tf.gather(flat_weights, expanded_indices)
  final_weights = tf.tensor_scatter_nd_update(new_weights, expanded_indices, keep_weights)  
  return final_weights

@tf.function
def select_weight(weights, my_id):
  return tf.reshape(tf.gather(weights,my_id),shape=[])

def build_fed_avg_process(
    total_clients: int,
    effective_num_clients: int,
    model_fn: ModelBuilder,
    client_optimizer_fn: OptimizerBuilder,
    client_lr: Union[float, LRScheduleFn] = 0.1,
    server_optimizer_fn: OptimizerBuilder = tf.keras.optimizers.SGD,
    server_lr: Union[float, LRScheduleFn] = 1.0,
    client_weight_fn: Optional[ClientWeightFn] = None,
    aggregation_process: Optional[measured_process.MeasuredProcess] = None,
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
    optimizer_variable_type = type_conversions.type_from_tensors(
        dummy_optimizer.variables())

  if aggregation_process is None:
    aggregation_process = build_stateless_mean(
        model_delta_type=model_weights_type.trainable)
    
  if not _is_valid_aggregation_process(aggregation_process):
    expected_signature = '(<state@S, input@C> -> <state@S, result@S, measurements@S>)'
    actual_signature = str(aggregation_process.next.type_signature)
    raise TypeError(
        f'aggregation_process type signature does not conform to expected '
        f'signature {expected_signature}.'
        ' Got: {t}'.format(t=actual_signature))
  
  initialize_computation = build_server_init_fn(
        model_fn = model_fn,
        effective_num_clients = effective_num_clients,
        # Initialize with the learning rate for round zero.
        server_optimizer_fn = lambda: server_optimizer_fn(server_lr_schedule(0)), 
        aggregation_process = aggregation_process)

  # server_state_type = initialize_computation.type_signature.result
  # model_weights_type = server_state_type.model
  round_num_type = tf.float32
  tf_dataset_type = tff.SequenceType(dummy_model.input_spec)
  model_input_type = tff.SequenceType(dummy_model.input_spec)

  client_losses_at_server_type = tff.TensorType(dtype=tf.float32, shape=[total_clients,1])
  clients_weights_at_server_type = tff.TensorType(dtype=tf.float32, shape=[total_clients,1])
  aggregation_state = aggregation_process.initialize.type_signature.result.member

  server_state_type = ServerState(
        model=model_weights_type,
        optimizer_state=optimizer_variable_type,
        round_num=round_num_type,
        effective_num_clients= tf.int32,
        delta_aggregate_state=aggregation_state,
        )

  #@tff.tf_computation(clients_weights_type)
  # def get_zero_weights_all_clients(weights):
  #   return tf.zeros_like(weights, dtype=tf.float32)

  ######################################################
  # def federated_output(local_outputs):
  #   return federated_aggregate_keras_metric(self.get_metrics(), local_outputs)

  # metric_finalizers = computations.federated_computation(
  #       federated_output, federated_local_outputs_type)


  single_id_type = tff.TensorType(dtype = tf.int32, shape = [1,1])
  @tff.tf_computation(model_input_type, model_weights_type, round_num_type, single_id_type)
  def client_update_fn(tf_dataset, initial_model_weights, round_num, client_id):
    client_lr = client_lr_schedule(round_num)
    client_optimizer = client_optimizer_fn(client_lr)
    client_update = create_client_update_fn()
    return client_update(model_fn(), tf_dataset, initial_model_weights,
                         client_optimizer,client_id,client_weight_fn)

  @tff.tf_computation(server_state_type, model_weights_type.trainable)
  def server_update_fn(server_state, model_delta):
    model = model_fn()
    server_lr = server_lr_schedule(server_state.round_num)
    server_optimizer = server_optimizer_fn(server_lr)
    # We initialize the server optimizer variables to avoid creating them
    # within the scope of the tf.function server_update.
    _initialize_optimizer_vars(model, server_optimizer)
    return server_update(model, server_optimizer, server_state, model_delta)

  id_type = tff.TensorType(shape=[1,1], dtype = tf.int32)

  @tff.tf_computation(clients_weights_at_server_type, id_type)
  def select_weight_fn(clients_weights, local_id):
    return select_weight(clients_weights, local_id)


  @tff.tf_computation(client_losses_at_server_type, clients_weights_at_server_type,tf.int32)
  def zero_small_loss_clients(losses_at_server, weights_at_server, effective_num_clients):
    """Receives losses and returns participating clients.

    Args:
      server_state: A `ServerState`.
      federated_dataset: A federated `tf.Dataset` with placement `tff.CLIENTS`.

    Returns:
      A tuple of updated `ServerState` and the result of
      `Union[tff.learning.models.VariableModel, tff.learning.models.FunctionalModel, tff.learning.models.ReconstructionModel].metric_finalizers`.
    """
    return redefine_client_weight( losses_at_server, weights_at_server, effective_num_clients)


  @tf.function
  def get_finalized_metrics(model_instance, federated_values_list):
      def apply_finalizers(unfinalized_metrics, finalizers):
          finalized_metrics = collections.OrderedDict()
          for metric_name, unfinalized_value in unfinalized_metrics.items():
              finalizer_function = finalizers[metric_name]
              finalized_value = finalizer_function(unfinalized_value)

              if metric_name not in finalized_metrics:
                  finalized_metrics[metric_name] = []

              finalized_metrics[metric_name].append(finalized_value)
          return finalized_metrics

      def aggregate_metrics(finalized_metrics):
          aggregated_metrics = collections.OrderedDict()
          for metric_name, values_list in finalized_metrics.items():
              aggregated_value = np.mean(values_list)  # You can use different aggregation methods
              aggregated_metrics[metric_name] = aggregated_value
          return aggregated_metrics

      # Get metric finalizers
      finalizers = model_instance.metric_finalizers()

      # Initialize a dictionary to store finalized metrics
      finalized_metrics = collections.OrderedDict()

      # Iterate through clients' federated values
      for federated_values in federated_values_list:
          # Get unfinalized metric values for the client
          unfinalized_metrics = model_instance.report_local_unfinalized_metrics()

          # Apply finalizers to compute finalized metric values
          client_finalized_metrics = apply_finalizers(unfinalized_metrics, finalizers)

          # Update the finalized metrics with client metrics
          for metric_name, client_metric in client_finalized_metrics.items():
              if metric_name not in finalized_metrics:
                  finalized_metrics[metric_name] = []

              finalized_metrics[metric_name].extend(client_metric)

      # Aggregate metrics across all clients
      aggregated_metrics = aggregate_metrics(finalized_metrics)

      return aggregated_metrics

  # @tff.tf_computation(client_losses_type)
  # def dataset_to_tensor_fn(dataset):
  #   return dataset_to_tensor(dataset)
  @tff.federated_computation(
      tff.FederatedType(server_state_type, tff.SERVER),
      tff.FederatedType(tf_dataset_type, tff.CLIENTS),
      tff.FederatedType(id_type, tff.CLIENTS))
  def run_one_round(server_state, federated_dataset, ids):
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
        (federated_dataset, client_model, client_round_num, ids))

    client_weight = client_outputs.client_weight
    client_id = client_outputs.client_id

    #LOSS SELECTION:
    # losses_at_server = tff.federated_collect(client_outputs.model_output)
    # weights_at_server = tff.federated_collect(client_weight)

    # Aggregate client weights and losses
    @tff.tf_computation
    def zeros_fn():
      return tf.zeros(shape=[total_clients,1] , dtype=tf.float32)

    zero = zeros_fn()

    at_server_type = tff.TensorType(shape=[total_clients,1],dtype=tf.float32)
    # list_type = tff.SequenceType( tff.TensorType(dtype=tf.float32))
    client_output_type = client_update_fn.type_signature.result

    @tff.tf_computation(at_server_type, client_output_type)
    def accumulate_weight(u,t):
      value = t.client_weight
      index = t.client_id
      new_u = tf.tensor_scatter_nd_update(u,index,value)  
      return new_u

    @tff.tf_computation(at_server_type, client_output_type)
    def accumulate_loss(u,t):
      value = tf.reshape(tf.math.reduce_sum(t.model_output.loss), shape = [1,1])
      index = t.client_id
      new_u = tf.tensor_scatter_nd_update(u,index,value)  
      return new_u

    # output_at_server= tff.federated_collect(client_outputs)
    
    weights_at_server = tff.federated_aggregate(client_outputs, zero, accumulate_weight)
    losses_at_server = tff.federated_aggregate(client_outputs, zero, accumulate_loss)
    #losses_at_server = tff.federated_aggregate(client_outputs.model_output, zero, accumulate, merge, report)

    selected_clients_weights = tff.federated_map(
      zero_small_loss_clients,
      (losses_at_server, weights_at_server, server_state.effective_num_clients))

    # selected_clients_weights_at_client = tff.federated_broadcast(selected_clients_weights)

    selected_clients_weights_broadcast = tff.federated_broadcast(selected_clients_weights)

    selected_clients_weights_at_client = tff.federated_map(select_weight_fn, (selected_clients_weights_broadcast, ids))

    aggregation_output = aggregation_process.next(
        server_state.delta_aggregate_state, client_outputs.weights_delta,
        selected_clients_weights_at_client)

    # model_delta = tff.federated_mean(
    #     client_outputs.weights_delta, weight=client_weight)

    server_state = tff.federated_map(server_update_fn,
                                     (server_state, aggregation_output.result))

    # Compute the finalized metrics using the get_finalized_metrics function, or use """dummy_model"""
    aggregated_outputs = get_finalized_metrics(dummy_model, federated_dataset)

    # Convert the finalized metrics into a FederatedType
    # aggregated_outputs = tff.federated_value(finalized_metrics, tff.SERVER)

    return server_state, aggregated_outputs

  # @tff.federated_computation
  # def initialize_fn():
  #   return tff.federated_value(server_init_tf(), tff.SERVER)

  return tff.templates.IterativeProcess(
      initialize_fn=initialize_computation, next_fn=run_one_round)
