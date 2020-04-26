# Copyright 2019 The Texar Authors. All Rights Reserved.
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
from _ast import If
"""
GPT2 decoders.
"""

import os
import sys
import json
import numpy as np

import tensorflow as tf
import texar.tf as tx
from texar.tf.module_base import ModuleBase
from custom_texar.custom_transformer_decoders import TransformerDecoder
from texar.tf.modules.embedders import PositionEmbedder, WordEmbedder


class GPT2Stack(ModuleBase):
    r"""Raw GPT2 Transformer for decoding sequences. Please see
    :class:`~texar.tf.modules.PretrainedGPT2Mixin` for a brief description
    of GPT2.

    This module basically stacks
    :class:`~texar.tf.modules.WordEmbedder`,
    :class:`~texar.tf.modules.PositionEmbedder`,
    :class:`~texar.tf.modules.TransformerDecoder`.

    This module supports the architecture first proposed
    in `(Radford et al.)` GPT2.

    Args:
        pretrained_model_name (optional): a `str`, the name
            of pre-trained model (e.g., ``gpt2-small``). Please refer to
            :class:`~texar.tf.modules.PretrainedGPT2Mixin` for
            all supported models.
            If `None`, the model name in :attr:`hparams` is used.
        cache_dir (optional): the path to a folder in which the
            pre-trained models will be cached. If `None` (default),
            a default directory (``texar_data`` folder under user's home
            directory) will be used.
        hparams (dict or HParams, optional): Hyperparameters. Missing
            hyperparameter will be set to default values. See
            :meth:`default_hparams` for the hyperparameter structure
            and default values.

    .. document private functions
    .. automethod:: _build
    """
    def __init__(self,
                 pretrained_model_name=None,
                 cache_dir=None,
                 hparams=None,
                 encode_mode=False):
        super().__init__(hparams=self.default_hparams())
        
        with tf.variable_scope(self.variable_scope):

            # Word embedding
            self.word_embedder = WordEmbedder(
                vocab_size=self._hparams.vocab_size,
                hparams=self._hparams.embed)

            # Position embedding
            self.position_embedder = PositionEmbedder(
                position_size=self._hparams.position_size,
                hparams=self._hparams.position_embed)

            # The GPT2 decoder (a TransformerDecoder)
            self.decoder = TransformerDecoder(
                vocab_size=self._hparams.vocab_size,
                output_layer=tf.transpose(self.word_embedder.embedding, (1, 0)),
                hparams=self._hparams.decoder,
                encode_mode=encode_mode)

    def embed_tokens(self, tokens, positions, mode=None):
        word_embeds = self.word_embedder(tokens, mode=mode)
        pos_embeds = self.position_embedder(positions, mode=mode)
        return word_embeds + pos_embeds
    
    # Expose GPT2 embeddings 
    def embeddings(self):
        return lambda tokens, positions, mode: self.embed_tokens(tokens, positions, mode)
    
    @staticmethod
    def default_hparams():
        """
        Remap the config file
        """
        input_json_path = "gpt2/gpt2-small/hparams.json"
        
        config_gpt = json.loads(open(input_json_path).read())
        configs = dict()
        configs["vocab_size"] = config_gpt["n_vocab"]
        configs["context_size"] = config_gpt["n_ctx"]
        configs["embedding_size"] = config_gpt["n_embd"]
        hidden_dim = config_gpt["n_embd"]
        configs["embed"] = {
            "dim": hidden_dim,
            "name": "word_embeddings"
        }
        configs["position_size"] = config_gpt["n_ctx"]
        configs["position_embed"] = {
            "dim": hidden_dim,
            "name": "position_embeddings"
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
    
        return configs
    
    def collect_trainable_variables(self):
        return tx.utils.collect_trainable_variables(
            [self.word_embedder,
             self.position_embedder,
             self.decoder]
            )
    
    def _build(self,
               decoding_strategy='train_greedy',
               inputs=None,
               memory=None,
               memory_sequence_length=None,
               memory_attention_bias=None,
               beam_width=None,
               length_penalty=0.,
               start_tokens=None,
               end_token=None,
               context=None,
               context_sequence_length=None,
               softmax_temperature=None,
               max_decoding_length=None,
               impute_finished=False,
               helper=None,
               mode=None,
               mle_context=None, # spamGAN MLE generator context
               sample_context=None # spamGAN sample generator context
               ):
        r"""Performs decoding. Has exact the same interfaces with
        :meth:`texar.tf.modules.TransformerDecoder._build` except inputs
        which is a tensor with shape `[batch_size, max_time]`. Please refer to
        it for the detailed usage.
        """
        if inputs is not None:
#             batch_size, max_time = inputs.shape.as_list()
            # Now support both static and dynamic shape
            batch_size, max_time = None, None
            
            if isinstance(inputs, tf.Tensor):
                batch_size = tf.shape(inputs)[0]
                max_time = tf.shape(inputs)[1] 
                
                try:
                    tf.shape(inputs)[2]
                except ValueError:
                    time = tf.expand_dims(tf.range(max_time), 0)
                    time = tf.broadcast_to(time, [batch_size, max_time])
                    inputs = self.embed_tokens(inputs, time, mode)
                else:
                    pass
            else:
                batch_size, max_time = inputs.shape.as_list()
            
            if mle_context is not None:
                inputs = tf.concat([inputs[:, :, :(inputs.shape[-1]-mle_context.shape[-1])], mle_context], axis = -1)
        
        outputs = self.decoder._build(
            decoding_strategy=decoding_strategy,
            inputs=inputs,
            memory=memory,
            memory_sequence_length=memory_sequence_length,
            memory_attention_bias=memory_attention_bias,
            beam_width=beam_width,
            length_penalty=length_penalty,
            start_tokens=start_tokens,
            end_token=end_token,
            context=context,
            context_sequence_length=context_sequence_length,
            softmax_temperature=softmax_temperature,
            max_decoding_length=max_decoding_length,
            impute_finished=impute_finished,
            embedding=lambda a, b, mode: self.embed_tokens(a, b, mode), # Introduce mode 
            helper=helper,
            mode=mode,
            sample_context=sample_context # spamGAN sample generator context
            )
        
        if not self._built:
            self._add_internal_trainable_variables()
            self._built = True
        
        print("outputs: {}".format(outputs))
        return outputs
    
