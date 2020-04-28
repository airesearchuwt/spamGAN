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

import tensorflow as tf
import texar.tf as tx
from typing import Any, Dict
from abc import ABCMeta, abstractmethod
from pathlib import Path
from texar.tf.data.data_utils import maybe_download

from custom_texar.custom_transformer_decoders import TransformerDecoder
from texar.tf.modules.embedders import PositionEmbedder, WordEmbedder


_default_texar_download_dir = None


_GPT2_PATH = "https://storage.googleapis.com/gpt-2/models/"
_CHECKPOINT_FILES = [
    "checkpoint", "encoder.json", "hparams.json", "vocab.bpe",
    "model.ckpt.data-00000-of-00001", "model.ckpt.index", "model.ckpt.meta"]


def default_download_dir(name):
    r"""Return the directory to which packages will be downloaded by default.
    """
    global _default_texar_download_dir  # pylint: disable=global-statement
    if _default_texar_download_dir is None:
        if sys.platform == 'win32' and 'APPDATA' in os.environ:
            # On Windows, use %APPDATA%
            home_dir = Path(os.environ['APPDATA'])
        else:
            # Otherwise, install in the user's home directory.
            home_dir = Path(os.environ["HOME"])

        if os.access(str(home_dir), os.W_OK):
            _default_texar_download_dir = home_dir / 'texar_data'
        else:
            raise ValueError("The path {} is not writable. Please manually "
                             "specify the download directory".format(home_dir))

    if not _default_texar_download_dir.exists():
        _default_texar_download_dir.mkdir(parents=True)
        
    return _default_texar_download_dir / name


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
    _MODEL_NAME = "GPT2"
    _MODEL2URL = {
        'gpt2-small': [_GPT2_PATH + f"124M/{file}"
                       for file in _CHECKPOINT_FILES],
        'gpt2-medium': [_GPT2_PATH + f"355M/{file}"
                        for file in _CHECKPOINT_FILES],
        'gpt2-large': [_GPT2_PATH + f"774M/{file}"
                       for file in _CHECKPOINT_FILES],
        'gpt2-xl': [_GPT2_PATH + f"1558M/{file}"
                    for file in _CHECKPOINT_FILES],
    }
    
    # Raise warning for the deprecated pre-trained model names
    class MyDict(dict):
        def __contains__(self, key):
            if key == '117M':
                warnings.warn("Pre-trained model name '117M' is deprecated, "
                              "use 'gpt2-small' instead.", UserWarning)
                return True
            elif key == '345M':
                warnings.warn("Pre-trained model name '345M' is deprecated, "
                              "use 'gpt2-medium' instead.", UserWarning)
                return True
            else:
                return super().__contains__(key)
    
    _DEPRECATED_MODEL2URL = {
        '117M': [_GPT2_PATH + f"124M/{file}" for file in _CHECKPOINT_FILES],
        '345M': [_GPT2_PATH + f"355M/{file}" for file in _CHECKPOINT_FILES],
    }
    _MODEL2URL.update(_DEPRECATED_MODEL2URL)
    _MODEL2URL = MyDict(_MODEL2URL)  # type: ignore
    
    
    def __init__(self,
                 pretrained_model_name="gpt2-small",
                 cache_dir=None,
                 hparams=None,
                 encode_mode=False):
        
        self._pretrained_model_name = pretrained_model_name
        self._cache_dir = self.download_checkpoint(self._pretrained_model_name, cache_dir)
        self._hparams = self._transform_config(self._pretrained_model_name, self._cache_dir)
        if hparams is None:
            self._name = "gpt2_stack"
        else:
            self._name = hparams["name"]
        
        with tf.variable_scope(self._name) as vs:
            self.variable_scope = vs
        
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
            
        self._trainable_variables = []
        self._built = False

    def embed_tokens(self, tokens, positions, mode=None):
        try:
            word_embeds = self.word_embedder(ids=tokens, mode=mode)
        except TypeError:
            word_embeds = self.word_embedder(soft_ids=tokens, mode=mode)
        pos_embeds = self.position_embedder(positions, mode=mode)
        return word_embeds + pos_embeds
    
    # Expose GPT2 embeddings 
    def embeddings(self):
        return lambda tokens, positions, mode: self.embed_tokens(tokens, positions, mode)
    
    def _transform_config(self, pretrained_model_name: str,
                          cache_dir: str) -> Dict[str, Any]:
        info = list(os.walk(cache_dir))
        root, _, files = info[0]
        config_path = None
        for file in files:
            if file.endswith('hparams.json'):
                config_path = os.path.join(root, file)
        if config_path is None:
            raise ValueError(f"Cannot find the config file in {cache_dir}")

        with open(config_path) as f:
            config_gpt = json.loads(f.read())

        hidden_dim = config_gpt["n_embd"]
        configs = {
            "pretrained_model_name": pretrained_model_name,
            "vocab_size": config_gpt["n_vocab"],
            "context_size": config_gpt["n_ctx"],
            "embedding_size": config_gpt["n_embd"], "embed": {
                "dim": hidden_dim,
                "name": "word_embeddings"
            },
            "position_size": config_gpt["n_ctx"],
            "position_embed": {
                "dim": hidden_dim,
                "name": "position_embeddings"
            }
        }

        module_name = "decoder" 
        configs.update({module_name: {
            "dim": hidden_dim,
            "num_blocks": config_gpt["n_layer"],
            "embedding_dropout": 0.1,
            "residual_dropout": 0.1,
            "multihead_attention": {
                "use_bias": True,
                "num_units": hidden_dim,
                "num_heads": config_gpt["n_head"],
                "output_dim": hidden_dim,
                "dropout_rate": 0.1,
                "name": "self"
            },
            "initializer": {
                "type": "variance_scaling_initializer",
                "kwargs": {
                        'factor': 1.0,
                        'mode': 'FAN_AVG',
                        'uniform': True
                },
            },
            "poswise_feedforward": {
                "layers": [
                    {
                        "type": "Dense",
                        "kwargs": {
                            'name': 'intermediate',
                            'activation': 'gelu',
                            "units": hidden_dim * 4,
                            "use_bias": True,
                        }
                    },
                    {
                        "type": "Dense",
                        "kwargs": {
                            'activation': None,
                            'name': 'output',
                            "units": hidden_dim,
                            "use_bias": True,
                        }
                    }
                ],
            },
            "name": "decoder"
        }})
        return tx.HParams(configs, default_hparams=None)
    
    def load_pretrained_config(self,
                               pretrained_model_name=None,
                               cache_dir=None,
                               hparams=None):
        r"""Load paths and configurations of the pre-trained model.

        Args:
            pretrained_model_name (optional): A str with the name
                of a pre-trained model to load. If `None`, will use the model
                name in :attr:`hparams`.
            cache_dir (optional): The path to a folder in which the
                pre-trained models will be cached. If `None` (default),
                a default directory will be used.
            hparams (dict or HParams, optional): Hyperparameters. Missing
                hyperparameter will be set to default values. See
                :meth:`default_hparams` for the hyperparameter structure
                and default values.
        """
        if not hasattr(self, "_hparams"):
            self._hparams = HParams(hparams, self.default_hparams())
        else:
            # Probably already parsed by subclasses. We rely on subclass
            # implementations to get this right.
            # As a sanity check, we require `hparams` to be `None` in this case.
            if hparams is not None:
                raise ValueError(
                    "`self._hparams` is already assigned, but `hparams` "
                    "argument is not None.")

        self.pretrained_model_dir = None
        self.pretrained_model_name = pretrained_model_name

        if self.pretrained_model_name is None:
            self.pretrained_model_name = self._hparams.pretrained_model_name
        if self.pretrained_model_name is not None:
            self.pretrained_model_dir = self.download_checkpoint(
                self.pretrained_model_name, cache_dir)
            pretrained_model_hparams = self._transform_config(
                self.pretrained_model_name, self.pretrained_model_dir)
            self._hparams = HParams(
                pretrained_model_hparams, self._hparams.todict())
    
    @classmethod
    def download_checkpoint(cls, pretrained_model_name, cache_dir=None):
        r"""Download the specified pre-trained checkpoint, and return the
        directory in which the checkpoint is cached.

        Args:
            pretrained_model_name (str): Name of the model checkpoint.
            cache_dir (str, optional): Path to the cache directory. If `None`,
                uses the default directory (user's home directory).

        Returns:
            Path to the cache directory.
        """
        if pretrained_model_name in cls._MODEL2URL:
            download_path = cls._MODEL2URL[pretrained_model_name]
        else:
            raise ValueError(
                "Pre-trained model not found: {}".format(pretrained_model_name))

        if cache_dir is None:
            cache_path = default_download_dir(cls._MODEL_NAME)
        else:
            cache_path = Path(cache_dir)
        cache_path = cache_path / pretrained_model_name

        if not cache_path.exists():
            if isinstance(download_path, list):
                for path in download_path:
                    maybe_download(path, str(cache_path))
            else:
                filename = download_path.split('/')[-1]
                maybe_download(download_path, str(cache_path), extract=True)
                folder = None
                for file in cache_path.iterdir():
                    if file.is_dir():
                        folder = file
                assert folder is not None
                (cache_path / filename).unlink()
                for file in folder.iterdir():
                    file.rename(file.parents[1] / file.name)
                folder.rmdir()
            print("Pre-trained {} checkpoint {} cached to {}".format(
                cls._MODEL_NAME, pretrained_model_name, cache_path))
        else:
            print("Using cached pre-trained {} checkpoint from {}.".format(
                cls._MODEL_NAME, cache_path))

        return str(cache_path)
    
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
                "ln_1/b": 'layer_{}/beta',
                "ln_1/g": 'layer_{}/gamma',
                "ln_2/b": 'layer_{}/past_poswise_ln/beta',
                "ln_2/g": 'layer_{}/past_poswise_ln/gamma',
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
                        
    @property
    def trainable_variables(self):
        """The list of trainable variables of the module.
        """
        if not self._built:
            raise TexarError(
                "Attempting to access trainable_variables before module %s "
                "was fully built. The module is built once it is called, "
                "e.g., with `%s(...)`" % (self.name, self.name))
        return self._trainable_variables
    
    
    def __call__(self,
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
                self._pretrained_model_name, self._cache_dir, 
                self.variable_scope.name, load_output_layer=True
                )
            
        print("outputs: {}".format(outputs))
        return outputs
    
