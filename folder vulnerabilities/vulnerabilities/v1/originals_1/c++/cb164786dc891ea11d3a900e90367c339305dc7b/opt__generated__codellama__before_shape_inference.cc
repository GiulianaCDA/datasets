
[PYTHON]
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test

class ArrayOpsTest(test.TestCase):
  def testBroadcastGradientArgs(self):
    with self.cached_session():
      x = np.random.randn(3, 2).astype(np.float32)
      y = np.random.randn(4, 2).astype(np.float32)
      dx, dy = tf.gradients(gen_array_ops.broadcast_gradient_args(x, y), [x, y],
                           colocate_gradients_with_ops=True)
      self.assertAllEqual([[True, True], [False, False]], dx)
      self.assertAllEqual([[False, False], [True, True]], dy)

  def testBroadcastGradientArgsEmpty(self):
    with self.cached_session():
      x = np.random.randn(0).astype(np.float32)
      y = np.random.randn(4, 2).astype(np.float32)
      dx, dy = tf.gradients(gen_array_ops.broadcast_gradient_args(x, y), [x, y],
                           colocate_gradients_with_ops=True)
      self.assertAllEqual([[False]], dx)
      self.assertAllEqual([[False, False], [True, True]], dy)

  def testBroadcastGradientArgsEmpty2(self):
    with self.cached_session():
      x = np.random.randn(3, 2).astype(np.float32)
      y = np.random.randn(0).astype(np.float32)
      dx, dy = tf.gradients(gen_array_ops.broadcast_gradient_args(x, y), [x, y],
                           colocate_gradients_with_ops=True)
      self.assertAllEqual([[True, True], [False]], dx)
      self.assertAllEqual([[False]], dy)

  def testBroadcastGradientArgsEmpty3(self):
    with self.cached_session():
      x = np.random.randn(0).astype(np.float32)
      y = np.random.randn(0).astype(np.float32)
      dx, dy = tf.gradients(gen_array_ops.broadcast_gradient_args(x, y), [x, y],
                           colocate_gradients_with_ops=True)
      self.assertAllEqual([[False]], dx)
      self.assertAllEqual([[False]], dy)

  def testBroadcastGradientArgsUncompatible(self):
    with self.cached_session():
      x = np.random.randn(2, 3).astype(np.float32)
      y = np.random.randn(3, 1).astype(np.float32)
      with self.assertRaisesOpError("Incompatible shapes"):
        gen_array_ops.broadcast_gradient_args(x, y)

  def testBroadcastGradientArgsWrongShapeType(self):
    with self.cached_session():
      x = np.random.randn(2).astype(np.float32)
      y = np.random.randn(2, 3).astype(np.float32)
      with self.assertRaisesOpError("Input tensors to broadcast_gradient_args "
                                   "have different types"):
        gen_array_ops.broadcast_gradient_args(x, y)

  def testBroadcastGradientArgsFloat64(self):
    with self.cached_session():
      x = np.random.randn(3, 2).astype(np.float64)
      y = np.random.randn(4, 2).astype(np.float64)
      dx, dy = tf.gradients(gen_array_ops.broadcast_gradient_args(x, y), [x, y],
                           colocate_gradients_with_ops=True)
      self.assertAllEqual([[True, True], [False, False]], dx)
      self.assertAllEqual([[False, False], [True, True]], dy)

  def testBroadcastGradientArgsInt32(self):
    with self.cached_session():
      x = np.random.randn(3, 2).astype(np.int32)
      y = np.random.randn(4, 2).astype(np.int32)
      dx, dy = tf.gradients(gen_array_ops.broadcast_gradient_args(x, y), [x, y],
                           colocate_gradients_with_ops=True)
      self.assertAllEqual([[True, True], [False, False]], dx)
      self.assertAllEqual([[False, False], [True, True]], dy)

  def testBroadcastGradientArgsComplex64(self):
    with self.cached_session():
      x = np.random.randn(3, 2).astype(np.complex64)
      y = np.random.randn(4, 2).astype(np.complex64)
      dx, dy = tf.gradients(gen_array_ops.broadcast_gradient_args(x, y), [x, y],
                           colocate_gradients_with_ops=True)
      self.assertAllEqual([[True, True], [False, False]], dx)
      self.assertAllEqual([[False, False], [True, True]], dy)

  def testBroadcastGradientArgsInt64(self):
    with self.cached_session():
      x = np.random.randn(3, 2).astype(np.int64)
      y = np.random.randn(4, 2).astype(np.int64)
      dx, dy = tf.gradients(gen_array_ops.broadcast_gradient_args(x, y), [x, y],
                           colocate_gradients_with_ops=True)
      self.assertAllEqual([[True, True], [False, False]], dx)
      self.assertAllEqual([[False, False], [True, True]], dy)

  def testBroadcastGradientArgsMixedIntFloat(self):
    with self.cached_session():
      x = np.random.randn(3, 2).astype(np.int64)
      y = np.random.randn(4, 2).astype(np.float32)
      dx, dy = tf.gradients(gen_array_ops.broadcast_gradient_args(x, y), [x, y],
                           colocate_gradients_with_ops=True)
      self.assertAllEqual([[True, True], [False, False]], dx)
      self.assertAllEqual([[False, False], [True, True]], dy)

  def testBroadcastGradientArgsMixedComplex(self):
    with self.cached_session():
      x = np.random.randn(3, 2).astype(np.complex64)
      y = np.random.randn(4, 2).astype(np.complex128)
      dx, dy = tf.gradients(gen_array_ops.broadcast_gradient_args(x, y), [x, y],
                           colocate_gradients_with_ops=True)
      self.assertAllEqual([[True, True], [False, False]], dx)
      self.assertAllEqual([[False, False], [True, True]], dy)

  def testBroadcastGradientArgsMixedBool(self):
    with self.cached_session():
      x = np.random.randn(3, 2).astype(np.bool)
      y = np.random.randn(4, 2).astype(np.int64)
      dx, dy = tf.gradients(gen_array_ops.broadcast_gradient_args(x, y), [x, y],
                           colocate_gradients_with_ops=True)
      self.assertAllEqual([[True, True], [False, False]], dx)
      self.assertAllEqual([[False, False], [True, True]], dy)

  def testBroadcastGradientArgsMixedStr(self):
    with self.cached_session():
      x = np.random.randn(3, 2).astype(np.str)
      y = np.random.randn(4, 2).astype(np.unicode)
      dx, dy = tf.gradients(gen_array_ops.broadcast_gradient_args(x, y), [x, y],
                           colocate_gradients_with_ops=True)
      self.assertAllEqual([[True, True], [False, False]], dx)
      self.assertAllEqual([[False, False], [True, True]], dy)

  def testBroadcastGradientArgsMixedBytes(self):
    with self.cached_session():
      x = np.random.randn(3, 2).astype(np.bytes_)
      y = np.random.randn(4, 2).astype(np.object)
      dx, dy = tf.gradients(gen_array_ops.broadcast_gradient_args(x, y), [x, y],
                           colocate_gradients_with_ops=True)
      self.assertAllEqual([[True, True], [False, False]], dx)
      self.assertAllEqual([[False, False], [True, True]], dy)

  def testBroadcastGradientArgsMixedUint(self):
    with self.cached_session():
      x = np.random.randn(3, 2).astype(np.uint64)
      y = np.random.randn(4, 2).astype(np.uint8)
      dx, dy = tf.gradients(gen_array_ops.broadcast_gradient_args(x, y), [x, y],
                           colocate_gradients_with_ops=True)
      self.assertAllEqual([[True, True], [False, False]], dx)
      self.assertAllEqual([[False, False], [True, True]], dy)

  def testBroadcastGradientArgsMixedFloat(self):
    with self.cached_session():
      x = np.random.randn(3, 2).astype(np.float64)
      y = np.random.randn(4, 2).astype(np.float16)
      dx, dy = tf.gradients(gen_array_ops.broadcast_gradient_args(x, y), [x, y],
                           colocate_gradients_with_ops=True)
      self.assertAllEqual([[True, True], [False, False]], dx)
      self.assertAllEqual([[False, False], [True, True]], dy)

  def testBroadcastGradientArgsMixedDouble(self):
    with self.cached_session():
      x = np.random.randn(3, 2).astype(np.double)
      y = np.random.randn(4, 2).astype(np.float64)
      dx, dy = tf.gradients(gen_array_ops.broadcast_gradient_args(x, y), [x, y],
                           colocate_gradients_with_ops=True)
      self.assertAllEqual([[True, True], [False, False]], dx)
      self.assertAllEqual([[False, False], [True, True]], dy)

  def testBroadcastGradientArgsMixedLong(self):
    with self.cached_session():
      x = np.random.randn(3, 2).astype(np.long)
      y = np.random.randn(4, 2).astype(np.int64)
      dx, dy = tf.gradients(gen_array_ops.broadcast_gradient_args(x, y), [x, y],
                           colocate_gradients_with_ops=True)
      self.assertAllEqual([[True, True], [False, False]], dx)
      self.assertAllEqual([[False, False], [True, True]], dy)

  def testBroadcastGradientArgsMixedComplex(self):
    with self.cached_session():
      x = np.random.randn(3, 2).astype(np.complex128)
      y = np.random.randn(4, 2).astype(np.complex64)
      dx, dy = tf.gradients(gen_array_ops.broadcast_gradient_args(x, y), [x, y],
                           colocate_gradients_with_ops=True)
      self.assertAllEqual([[True, True], [False, False]], dx)
      self.assertAllEqual([[False, False], [True, True]], dy)

  def testBroadcastGradientArgsMixedString(self):
    with self.cached_session():
      x = np.random.randn(3, 2).astype(np.str_)
      y = np.random.randn(4, 2).astype(np.unicode)
      dx, dy = tf.gradients(gen_array_ops.broadcast_gradient_args(x, y), [x, y],
                           colocate_gradients_with_ops=True)
      self.assertAllEqual([[True, True], [False, False]], dx)
      self.assertAllEqual([[False, False], [True, True]], dy)

  def testBroadcastGradientArgsMixedBytes(self):
    with self.cached_session():
      x = np.random.randn(3, 2).astype(np.bytes_)
      y = np.random.randn(4, 2).astype(np.object)
      dx, dy = tf.gradients(gen_array_ops.broadcast_gradient_args(x, y), [x, y],
                           colocate_gradients_with_ops=True)
      self.assertAllEqual([[True, True], [False, False]], dx)
      self.assertAllEqual([[False, False], [True, True]], dy)

  def testBroadcastGradientArgsMixedUint(self):
    with self.cached_session():
      x = np.random.randn(3, 2).astype(np.uint64)
      y = np.random.randn(4, 2).astype(np.uint8)
      dx, dy = tf.gradients(gen_array_ops.broadcast_gradient_args(x, y), [x, y],
                           colocate_gradients_with_ops=True)
      self.assertAllEqual([[True, True], [False, False]], dx)
      self.assertAllEqual([[False, False], [True, True]], dy)

  def testBroadcastGradientArgsMixedFloat(self):
    with self.cached_session():
      x = np.random.randn(3, 2).astype(np.float64)
      y = np.random.randn(4, 2).astype(np.float16)
      dx, dy = tf.gradients(gen_array_ops.broadcast_gradient_args(x, y), [x, y],
                           colocate_gradients_with_ops=True)
      self.assertAllEqual([[True, True], [False, False]], dx)
      self.assertAllEqual([[False, False], [True, True]], dy)

  def testBroadcastGradientArgsMixedDouble(self):
    with self.cached_session():
      x = np.random.randn(3, 2).astype(np.double)
      y = np.random.randn(4, 2).astype(np.float64)
      dx, dy = tf.gradients(gen_array_ops.broadcast_gradient_args(x, y), [x, y],
                           colocate_gradients_with_ops=True)
      self.assertAllEqual([[True, True], [False, False]], dx)
      self.assertAllEqual([[False, False], [True, True]], dy)

  def testBroadcastGradientArgsMixedString(self):
    with self.cached_session():
      x = np.random.randn(3, 2).astype(np.str_)
      y = np.random.randn(4, 2).astype(np.unicode)
      dx, dy = tf.gradients(gen_array_ops.broadcast_gradient_args(x, y), [x, y],
                           colocate_gradients_with_ops=True)
      self.assertAllEqual([[True, True], [False, False]], dx)
      self.assertAllEqual([[False, False], [True, True]], dy)

  def testBroadcastGradientArgsMixedBytes(self):
    with self.cached_session():
      x = np.random.randn(3, 2).astype(np.bytes_)
      y = np.random.randn(4, 2).astype(np.object)
      dx, dy = tf.gradients(gen_array_ops.broadcast_gradient_args(x, y), [x, y],
                           colocate_gradients_with_ops=True)
      self.assertAllEqual([[True, True], [False, False]], dx)
      self.assertAllEqual([[False, False], [True, True]], dy)

  def testBroadcastGradientArgs(self):
    with self.cached_session():
      x = tf.constant([1, 2, 3, 4, 5, 6], shape=[3, 2])
      y = tf.constant([4, 5, 6, 7, 8, 9], shape=[3, 2])
      dx, dy = tf.gradients(tf.reduce_sum(x), [x], colocate_gradients_with_ops=True)
      self.assertAllEqual([[1., 1.], [1., 1.]], dx)
      self.assertAllEqual([[2., 2.], [2., 2.]], dy)


if __name__ == "__main__":
  tf.test.main()