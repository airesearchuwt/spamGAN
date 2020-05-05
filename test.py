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
#     x = tf.constant([[4], [3], [1], [7]])
#     y = tf.constant([9, 10, 11, 12])
#     shape = y.shape
#     
#     batch_size = 2
#     time = 16
#     
#     sample_ids = tf.constant([1, -1])
#     sample_ids_after = tf.reshape(sample_ids, [tf.shape(sample_ids)[0], -1])
#     
#     where_sampling = math_ops.cast(
#         array_ops.where(sample_ids > -1), dtypes.int32)
#     where_not_sampling = math_ops.cast(
#         array_ops.where(sample_ids <= -1), dtypes.int32)
#     
#     times = tf.ones(batch_size, dtype=tf.int32) * (time + 1)
#     
#     with tf.compat.v1.Session() as sess:
#         print("previous: {}".format(sample_ids))
#         print("after: {}".format(sample_ids_after))
#         print(where_sampling.eval())
#         print(where_not_sampling.eval())
#         print(array_ops.gather_nd(sample_ids, where_sampling).eval())
#         print(array_ops.gather_nd(sample_ids, where_not_sampling).eval())
#         print("times: {}".format(times.eval()))
#         print(array_ops.scatter_nd(x, y, shape).eval())

    t = "Ġcomplimentary Ġexecute Ġfor Ġsupposed Ġclassy Ġto Ġstay Ġhave Ġyour Ġissue Ġ, Ġthe Ġtime Ġstock Ġseem Ġ. Ġearlier Ġthe Ġarrogant Ġtea Ġrepeatedly Ġfinally Ġ' Ġt Ġmake Ġturned Ġ. Ġ. Ġalmost Ġthey Ġthe $ Ġ. Ġcenter Ġthanking Ġpool Ġ' Ġbroken Ġbucks Ġago Ġday $ Ġnot Ġthem Ġcould Ġwhy Ġroom Ġreally Ġof Ġtwo Ġupgraded Ġtheir Ġgive Ġday Ġ? Ġthey Ġperk Ġhe Ġtravel Ġyou Ġmy Ġused Ġminutes Ġduring Ġlaughable Ġone Ġraised Ġyou Ġis Ġhim Ġwill Ġonly Ġvery Ġhesitate Ġupset Ġalone Ġthe Ġ' $ Ġto Ġthe Ġany Ġbugs Ġi Ġwould Ġkeep Ġyou Ġme Ġforeign Ġwould Ġhad Ġwoke Ġit Ġ. Ġsomehow Ġsays Ġincredibly Ġrooms Ġwas Ġnotch Ġexcellent ĠN Ġniche Ġcouple Ġ. Ġstrange Ġjust Ġ. Ġexcept Ġsize Ġhad Ġkind Ġabout Ġeternity Ġinternet Ġmistakes Ġtop ĠN Ġstain Ġnight Ġcome Ġused Ġto Ġyou ĠN Ġfirst Ġeverything Ġthe Ġis Ġquality Ġworst Ġ, Ġ. Ġgreat Ġi $ Ġexcellent Ġvery Ġon Ġto Ġ' Ġlobby Ġhas Ġyou Ġup Ġsince Ġ' Ġa Ġthe Ġlooking Ġquality Ġ. Ġthe Ġmajor Ġ. Ġlocation Ġ, $ Ġneed Ġavoid Ġrolled Ġtake Ġthat Ġwouldn Ġstaff Ġhas Ġa Ġhotel Ġchlorine Ġthe Ġcenter Ġthe Ġblock Ġthree Ġabout Ġ Ġfare Ġaverage Ġwould Ġme Ġmeet Ġyou Ġhave Ġa Ġto Ġmanagement Ġmost Ġlike Ġid Ġapology Ġwell Ġnot Ġno Ġfuture Ġ, Ġoh Ġfeel Ġthe Ġmy Ġmy Ġwill Ġsugar Ġto Ġpay Ġdots Ġinappropriate Ġ! Ġ( Ġalmost Ġsuggest Ġone $ <|endoftext|> <|padding|> Ġconvenience Ġmass Ġtour Ġannounced Ġinjury Ġsince Ġ. Ġdate Ġwater Ġsuit Ġstation Ġ, Ġsun Ġof Ġwith Ġcalm Ġof Ġgrocery Ġstation Ġthe Ġbonuses $ Ġcloset Ġ! Ġand Ġ. Ġsun Ġ-- Ġcard Ġwith Ġ) Ġmakeshift Ġmagical Ġmovies Ġcity Ġtennis Ġtunnel Ġand Ġpolicy Ġred Ġ Ġspeaker"
    print(len(t.split(" ")))
    
    
