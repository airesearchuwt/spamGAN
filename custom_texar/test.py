import tensorflow as tf
import texar.tf as tx

import os
import sys
import json
import numpy as np



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
            

def transform_gpt2_to_texar_config(input_json_path):
        """
        Remap the config file
        """
        config_gpt = json.loads(open(input_json_path).read())
        configs = dict()
        configs["vocab_size"] = config_gpt["n_vocab"]
        configs["context_size"] = config_gpt["n_ctx"]
        configs["embedding_size"] = config_gpt["n_embd"]
        hidden_dim = config_gpt["n_embd"]
        configs["embed"] = {
            "dim": hidden_dim,
        }
        configs["position_size"] = config_gpt["n_ctx"]
        configs["pos_embed"] = {
            "dim": hidden_dim
        }
        configs["decoder"] = {
            "dim": hidden_dim,
            "embedding_dropout": 0.1,
            "residual_dropout": 0.1,
            "num_blocks": config_gpt["n_layer"],
            "multihead_attention": {
                "use_bias": True,
                "num_units": hidden_dim,
                "num_heads": config_gpt["n_head"],
                "output_dim": hidden_dim,
                'dropout_rate': 0.1,
            },
            "initializer": {
                "type": "variance_scaling_initializer",
                "kwargs": {
                    "scale": 1.0,
                    "mode": "fan_avg",
                    "distribution": "uniform",
                },
            },
            "poswise_feedforward": {
                "layers": [
                    {
                        "type": "Dense",
                        "kwargs": {
                            "name": "conv1",
                            "units": hidden_dim * 4,
                            "activation": "gelu",
                            "use_bias": True,
                        }
                    },
                    {
                        "type": "Dense",
                        "kwargs": {
                            "name": "conv2",
                            "units": hidden_dim,
                            "use_bias": True,
                        }
                    }
                ],
                "name": "ffn",
            },
        }
        configs["name"] =  "gpt2_stack"
        
        return tx.HParams(configs, default_hparams=None)


if __name__ == "__main__":
    input_json_path = "../gpt2/gpt2-small/hparams.json"
    h = transform_gpt2_to_texar_config(input_json_path)
    print(h)
    
    