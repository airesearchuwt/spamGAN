import tensorflow as tf
from tensorflow.python.ops import array_ops, math_ops
from tensorflow.python.framework import dtypes
import json


def test_random_sample():
    samples = tf.random.categorical(
        tf.math.log([[[0.05, 0.2, 0.15, 0.5, 0.1]]]),
        1)
    
    with tf.compat.v1.Session() as sess:
        first = 0
        second = 0
        third = 0
        fourth = 0
        fifth = 0
        
        for i in range(1000):
            for seq in samples.eval():
                for index in seq:
                    if index == 0:
                        first = first + 1
                    elif index == 1:
                        second = second + 1
                    elif index == 2:
                        third = third + 1
                    elif index == 3:
                        fourth = fourth + 1
                    elif index == 4:
                        fifth = fifth + 1
        
        print("Index 0: {} times".format(first))
        print("Index 1: {} times".format(second))
        print("Index 2: {} times".format(third))
        print("Index 3: {} times".format(fourth))
        print("Index 4: {} times".format(fifth))
            


if __name__ == "__main__":
    x = tf.constant([[4], [3], [1], [7]])
    y = tf.constant([9, 10, 11, 12])
    shape = y.shape
    
    batch_size = 2
    time = 16
    
    sample_ids = tf.constant([1, -1])
    sample_ids_after = tf.reshape(sample_ids, [tf.shape(sample_ids)[0], -1])
    
    where_sampling = math_ops.cast(
        array_ops.where(sample_ids > -1), dtypes.int32)
    where_not_sampling = math_ops.cast(
        array_ops.where(sample_ids <= -1), dtypes.int32)
    
    times = tf.ones(batch_size, dtype=tf.int32) * (time + 1)
    
    with tf.compat.v1.Session() as sess:
        print("previous: {}".format(sample_ids))
        print("after: {}".format(sample_ids_after))
        print(where_sampling.eval())
        print(where_not_sampling.eval())
        print(array_ops.gather_nd(sample_ids, where_sampling).eval())
        print(array_ops.gather_nd(sample_ids, where_not_sampling).eval())
        print("times: {}".format(times.eval()))
#         print(array_ops.scatter_nd(x, y, shape).eval())
         
    
    
