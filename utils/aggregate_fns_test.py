# Copyright 2019, Google LLC.  #
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

import collections

import tensorflow as tf
import tensorflow_federated as tff

from utils import aggregate_fns


def create_weights_delta(input_size=2, hidden_size=5, constant=0):
  """Returns deterministic weights delta for a linear model."""
  kernel = constant + tf.reshape(
      tf.range(input_size * hidden_size, dtype=tf.float32),
      [input_size, hidden_size])
  bias = constant + tf.range(hidden_size, dtype=tf.float32)
  return collections.OrderedDict([('dense/kernel', kernel),
                                  ('dense/bias', bias)])


class FixedClipNormProcessTest(tf.test.TestCase):

  def test_clip_by_global_norm(self):
    clip_norm = 20.0
    test_deltas = [create_weights_delta(), create_weights_delta(constant=10)]
    update_type = tff.types.type_from_tensors(test_deltas[0])
    aggregate_fn = aggregate_fns.build_fixed_clip_norm_mean_process(
        clip_norm=clip_norm, model_update_type=update_type)

    self.assertTrue(
        aggregate_fn.next.type_signature.is_equivalent_to(
            tff.FunctionType(
                parameter=collections.OrderedDict(
                    state=tff.FederatedType((), tff.SERVER),
                    deltas=tff.FederatedType(update_type, tff.CLIENTS),
                    weights=tff.FederatedType(tf.float32, tff.CLIENTS),
                ),
                result=tff.templates.MeasuredProcessOutput(
                    state=tff.FederatedType((), tff.SERVER),
                    result=tff.FederatedType(update_type, tff.SERVER),
                    measurements=tff.FederatedType(
                        aggregate_fns.NormClippedAggregationMetrics(
                            max_global_norm=tf.float32, num_clipped=tf.int32),
                        tff.SERVER)),
            )))

    state = aggregate_fn.initialize()
    weights = [1., 1.]
    output = aggregate_fn.next(state, test_deltas, weights)

    expected_clipped = []
    for delta in test_deltas:
      clipped, _ = tf.clip_by_global_norm(tf.nest.flatten(delta), clip_norm)
      expected_clipped.append(tf.nest.pack_sequence_as(delta, clipped))
    expected_mean = tf.nest.map_structure(lambda a, b: (a + b) / 2,
                                          *expected_clipped)
    self.assertAllClose(expected_mean, output.result)

    # Global l2 norms [17.74824, 53.99074].
    metrics = output.measurements
    self.assertAlmostEqual(metrics.max_global_norm, 53.99074, places=5)
    self.assertEqual(metrics.num_clipped, 1)


if __name__ == '__main__':
  tf.test.main()
