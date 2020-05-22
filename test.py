import tensorflow as tf
from tensorflow.python.ops import array_ops, math_ops
from tensorflow.python.framework import dtypes
import json
from numpy import random

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
            

def test_random_traitor(times):
    cnt_0 = 0
    cnt_1 = 0
    
    for i in range(times):
        t = random.choice(a=[False, True], size=1, p=[0.3, 0.7])
        if t == 0:
            cnt_0 = cnt_0 + 1
        else:
            cnt_1 = cnt_1 + 1
            
    print("times of choosing 0: {}".format(cnt_0))
    print("times of choosing 1: {}".format(cnt_1))
            
        

if __name__ == "__main__":
#     all_data_labels = tf.constant(
#         [[1], [0], [1], [0], 
#         [-1], [-1], [-1], [-1]]
#         )
    all_data_labels = tf.constant(
        [[1], [0], [1], [0]]
        )
    labeled = tf.logical_not(tf.equal(all_data_labels, -1))
    any_labeled = tf.reduce_any(labeled)
    
    reclass_unlab = tf.zeros_like(all_data_labels)
    true_classes = tf.where(labeled, all_data_labels, reclass_unlab)
     
    with tf.compat.v1.Session() as sess:
        print("all_data_labels: {}".format(all_data_labels.eval()))
        print("labeled: {}".format(labeled.eval()))
        print("reclass_unlab: {}".format(reclass_unlab.eval()))
        print("true_classes: {}".format(true_classes.eval()))
        print("true_classes shape: {}".format(true_classes.shape))
        
        
