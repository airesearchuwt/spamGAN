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
import collections
import re
import warnings

from typing import Any, Dict

import tensorflow as tf
import texar.tf as tx
from custom_texar.custom_transformer_decoders import TransformerDecoder
from texar.tf.modules.embedders import PositionEmbedder, WordEmbedder


class GPT2Stack():
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
        if pretrained_model_name is None:
            self.pretrained_model_name = "gpt2-small"
        else:
            self.pretrained_model_name = pretrained_model_name
        if cache_dir is None:
            self.cache_dir = os.path.abspath(
                os.path.join("./gpt2", self.pretrained_model_name)
                )
        else:
            self.cache_dir = cache_dir
        self._hparams = self.default_hparams(self.cache_dir)
        self.variable_scope = "gpt2_stack"
        
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
            
        self._built = False

    def embed_tokens(self, tokens, positions, mode=None):
        word_embeds = self.word_embedder(tokens, mode=mode)
        pos_embeds = self.position_embedder(positions, mode=mode)
        return word_embeds + pos_embeds
    
    # Expose GPT2 embeddings 
    def embeddings(self):
        return lambda tokens, positions, mode: self.embed_tokens(tokens, positions, mode)
    
    @staticmethod
    def default_hparams(cache_dir):
        """
        Remap the config file
        """
        input_json_path = os.path.abspath(
            os.path.join(cache_dir, "hparams.json")
            )
        
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
    
        return tx.HParams(configs, default_hparams=None)
    
    def _init_from_checkpoint(self, pretrained_model_name, cache_dir,
                          scope_name, load_output_layer=True, **kwargs):
        r"""Initialize model parameters from weights stored in the pre-trained
        checkpoint.

        Args:
            cache_dir (str): Path to the cache directory.
            scope_name (str): Scope name of the model.
            load_output_layer (bool): If `False`, will not load weights of the
                output layer. Set this argument to `False` when loading weights
                into a GPT2 encoder. Defaults to `True`.
        """
        init_checkpoint = os.path.abspath(os.path.join(cache_dir,
                                                       'model.ckpt'))
        ckpt = tf.train.load_checkpoint(init_checkpoint)
        ckpt_params = {key: ckpt.get_tensor(key) for key in
                       ckpt.get_variable_to_shape_map().keys()}

        tvars = tf.trainable_variables()
        name_to_variable = collections.OrderedDict()
        for var in tvars:
            name = var.name
            m = re.match("^(.*):\\d+$", name)
            if m is not None:
                name = m.group(1)
            name_to_variable[name] = var

        if load_output_layer:
            global_tensor_map = {
                'model/wte': scope_name + '/word_embeddings/w',
                'model/wpe': scope_name + '/position_embeddings/w',
                'model/ln_f/b': scope_name + '/decoder/beta',
                'model/ln_f/g': scope_name + '/decoder/gamma',
            }

            layer_tensor_map = {
                "ln_1/b": scope_name + '/layer_{}/beta',
                "ln_1/g": scope_name + '/layer_{}/gamma',
                "ln_2/b": scope_name + '/layer_{}/past_poswise_ln/beta',
                "ln_2/g": scope_name + '/layer_{}/past_poswise_ln/gamma',
                "mlp/c_fc/b": scope_name + '/decoder/layer_{}'
                                           '/ffn/intermediate/bias',
                "mlp/c_fc/w": scope_name + '/decoder/layer_{}'
                                           '/ffn/intermediate/kernel',
                "mlp/c_proj/b": scope_name + '/decoder/layer_{}/ffn/output/'
                                             'bias',
                "mlp/c_proj/w": scope_name + '/decoder/layer_{}/ffn/output/'
                                             'kernel',
                "attn/c_attn/b": None,
                "attn/c_attn/w": None,
                "attn/c_proj/b": scope_name + '/decoder/layer_{}'
                                              '/self_attention/self/output/'
                                              'bias',
                "attn/c_proj/w": scope_name + '/decoder/layer_{}'
                                              '/self_attention/self/output/'
                                              'kernel',
            }
        else:
            global_tensor_map = {
                'model/wte': scope_name + '/word_embeddings/w',
                'model/wpe': scope_name + '/position_embeddings/w',
                'model/ln_f/b': scope_name + '/encoder/LayerNorm/beta',
                'model/ln_f/g': scope_name + '/encoder/LayerNorm/gamma',
            }

            layer_tensor_map = {
                "ln_1/b": scope_name + '/encoder/layer_{}/LayerNorm/beta',
                "ln_1/g": scope_name + '/encoder/layer_{}/LayerNorm/gamma',
                "ln_2/b": scope_name + '/encoder/layer_{}/output/'
                                       'LayerNorm/beta',
                "ln_2/g": scope_name + '/encoder/layer_{}/output/'
                                       'LayerNorm/gamma',
                "mlp/c_fc/b": scope_name + '/encoder/layer_{}'
                                           '/ffn/intermediate/bias',
                "mlp/c_fc/w": scope_name + '/encoder/layer_{}'
                                           '/ffn/intermediate/kernel',
                "mlp/c_proj/b": scope_name + '/encoder/layer_{}/ffn/output/'
                                             'bias',
                "mlp/c_proj/w": scope_name + '/encoder/layer_{}/ffn/output/'
                                             'kernel',
                "attn/c_attn/b": None,
                "attn/c_attn/w": None,
                "attn/c_proj/b": scope_name + '/encoder/layer_{}'
                                              '/attention/self/output/bias',
                "attn/c_proj/w": scope_name + '/encoder/layer_{}'
                                              '/attention/self/output/kernel',
            }

        for name, array in ckpt_params.items():
            if name in global_tensor_map:
                v_name = global_tensor_map[name]
                pointer = name_to_variable[v_name]
                pointer._initializer_op = tf.assign(pointer._variable, array)
            else:
                name_tmp = name.split("/")
                layer_no = name_tmp[1][1:]
                name = "/".join(name_tmp[2:])

                if name in layer_tensor_map:
                    if name == "attn/c_attn/b":
                        if load_output_layer:
                            K = name_to_variable[
                                scope_name + '/decoder/layer_' + layer_no +
                                '/self_attention/self/key/bias']
                            Q = name_to_variable[
                                scope_name + '/decoder/layer_' + layer_no +
                                '/self_attention/self/query/bias']
                            V = name_to_variable[
                                scope_name + '/decoder/layer_' + layer_no +
                                '/self_attention/self/value/bias']
                        else:
                            K = name_to_variable[
                                scope_name + '/encoder/layer_' + layer_no +
                                '/attention/self/key/bias']
                            Q = name_to_variable[
                                scope_name + '/encoder/layer_' + layer_no +
                                '/attention/self/query/bias']
                            V = name_to_variable[
                                scope_name + '/encoder/layer_' + layer_no +
                                '/attention/self/value/bias']

                        index_d = array.shape[-1] // 3

                        Q_w = array[:index_d]
                        K_w = array[index_d: 2 * index_d]
                        V_w = array[2 * index_d:]

                        K._initializer_op = tf.assign(K._variable, K_w)
                        Q._initializer_op = tf.assign(Q._variable, Q_w)
                        V._initializer_op = tf.assign(V._variable, V_w)
                    elif name == "attn/c_attn/w":
                        if load_output_layer:
                            K = name_to_variable[
                                scope_name + '/decoder/layer_' + layer_no +
                                '/self_attention/self/key/kernel']
                            Q = name_to_variable[
                                scope_name + '/decoder/layer_' + layer_no +
                                '/self_attention/self/query/kernel']
                            V = name_to_variable[
                                scope_name + '/decoder/layer_' + layer_no +
                                '/self_attention/self/value/kernel']
                        else:
                            K = name_to_variable[
                                scope_name + '/encoder/layer_' + layer_no +
                                '/attention/self/key/kernel']
                            Q = name_to_variable[
                                scope_name + '/encoder/layer_' + layer_no +
                                '/attention/self/query/kernel']
                            V = name_to_variable[
                                scope_name + '/encoder/layer_' + layer_no +
                                '/attention/self/value/kernel']

                        index_d = array.shape[-1] // 3

                        Q_w = np.transpose(array[0, :, :index_d])
                        K_w = np.transpose(array[0, :, index_d: 2 * index_d])
                        V_w = np.transpose(array[0, :, 2 * index_d:])

                        K._initializer_op = tf.assign(K._variable, K_w)
                        Q._initializer_op = tf.assign(Q._variable, Q_w)
                        V._initializer_op = tf.assign(V._variable, V_w)
                    elif (name == "attn/c_proj/w" or name == "mlp/c_fc/w" or
                          name == "mlp/c_proj/w"):
                        v_name = layer_tensor_map[name]
                        pointer = name_to_variable[v_name.format(layer_no)]
                        pointer._initializer_op = tf.assign(pointer._variable,
                                                            array[0])
                    else:
                        v_name = layer_tensor_map[name]
                        pointer = name_to_variable[v_name.format(layer_no)]
                        pointer._initializer_op = tf.assign(pointer._variable,
                                                            array)
    
    def _add_internal_trainable_variables(self):  # pylint: disable=invalid-name
        """Collects trainable variables constructured internally in this module.

        This is typically called at the end of `_build()` where all necessary
        trainable variables have been constructed.
        """
        scope_name = self.variable_scope.name
        # Escape to handle possible "." characters in the name.
        # Append a slash to the end to avoid searching scopes that have this
        # scope name as a prefix.
        scope_name = re.escape(scope_name) + "/"
        internal_trainable_variables = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope_name)
        self._add_trainable_variable(internal_trainable_variables)

    def _add_trainable_variable(self, variable):
        """Adds a trainable variable to the trainable variable list of the
        module.

        Args:
            variable: a (list of) trainable variable(s) constructed either
                internally in the module or constructured outside but used
                inside the module.
        """
        if isinstance(variable, (list, tuple)):
            for var in variable:
                self._add_trainable_variable(var)
        else:
            if variable not in self._trainable_variables:
                self._trainable_variables.append(variable)
                        
#     def collect_trainable_variables(self):
#         return tx.utils.collect_trainable_variables(
#             [self.word_embedder,
#              self.position_embedder,
#              self.decoder]
#             )
    
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
        
        if self._built is False:
            self._add_internal_trainable_variables()
            self._built = True
            self._init_from_checkpoint(
                self.pretrained_model_name, self.cache_dir,
                self.variable_scope.name, load_output_layer=True, **kwargs
                )
        
        print("outputs: {}".format(outputs))
        return outputs
    
