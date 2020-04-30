## MODIFIED TO ALLOW CLASS EMBEDDING APPENDING

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=no-name-in-module

import abc

import six

import tensorflow as tf
from tensorflow.contrib.seq2seq.python.ops import decoder
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow_probability import distributions as tfpd
from tensorflow.python.util import nest

import texar.tf as tx
from texar.tf.utils.shapes import shape_list
from texar.tf.utils.utils import get_args

from texar.tf.modules.embedders.embedder_utils import soft_embedding_lookup
from texar.tf.utils import utils


_transpose_batch_time = decoder._transpose_batch_time  # pylint: disable=protected-access


def _unstack_ta(inp):
    return tensor_array_ops.TensorArray(
        dtype=inp.dtype, size=array_ops.shape(inp)[0],
        element_shape=inp.get_shape()[1:]).unstack(inp)


class Helper(object):
  """Interface for implementing sampling in seq2seq decoders.
  Helper instances are used by `BasicDecoder`.
  """

  def batch_size(self):
    """Batch size of tensor returned by `sample`.
    Returns a scalar int32 tensor.
    """
    raise NotImplementedError("batch_size has not been implemented")

  def sample_ids_shape(self):
    """Shape of tensor returned by `sample`, excluding the batch dimension.
    Returns a `TensorShape`.
    """
    raise NotImplementedError("sample_ids_shape has not been implemented")

  def sample_ids_dtype(self):
    """DType of tensor returned by `sample`.
    Returns a DType.
    """
    raise NotImplementedError("sample_ids_dtype has not been implemented")

  def initialize(self, name=None):
    """Returns `(initial_finished, initial_inputs)`."""
    pass

  def sample(self, time, outputs, state, name=None):
    """Returns `sample_ids`."""
    pass

  def next_inputs(self, time, outputs, state, sample_ids, name=None):
    """Returns `(finished, next_inputs, next_state)`."""
    pass

class ContextGreedyEmbeddingHelper(Helper):
  """A helper for use during inference.
  Uses the argmax of the output (treated as logits) and passes the
  result through an embedding layer to get the next input.
  """

  def __init__(self, embedding, context, start_tokens, end_token):
    """Initializer.
    Args:
      embedding: A callable that takes a vector tensor of `ids` (argmax ids),
        or the `params` argument for `embedding_lookup`. The returned tensor
        will be passed to the decoder input.
      start_tokens: `int32` vector shaped `[batch_size]`, the start tokens.
      end_token: `int32` scalar, the token that marks end of decoding.
    Raises:
      ValueError: if `start_tokens` is not a 1D tensor or `end_token` is not a
        scalar.
    """
    if isinstance(embedding, tx.modules.EmbedderBase):
        embedding = embedding.embedding

    if callable(embedding):
        raise ValueError("`embedding` must be an embedding tensor or an "
                         "instance of subclass of `EmbedderBase`.")
    else:
        self._embedding = embedding
        self._embedding_fn = (
            lambda ids: tf.nn.embedding_lookup(embedding, ids))
    self.context = context
    self._start_tokens = tf.convert_to_tensor(
        start_tokens, dtype=tf.int32, name="start_tokens")
    self._end_token = tf.convert_to_tensor(
        end_token, dtype=tf.int32, name="end_token")
    if self._start_tokens.get_shape().ndims != 1:
      raise ValueError("start_tokens must be a vector")
    self._batch_size = tf.size(start_tokens)
    if self._end_token.get_shape().ndims != 0:
      raise ValueError("end_token must be a scalar")
    self._start_inputs = self._embedding_fn(self._start_tokens)
    print("self._start_inputs: {}".format(self._start_inputs))
    print("self.context: {}".format(self.context))
    self._start_inputs = tf.concat([self._start_inputs, self.context], axis = -1)

  @property
  def batch_size(self):
    return self._batch_size

  @property
  def sample_ids_shape(self):
    return tf.TensorShape([])

  @property
  def sample_ids_dtype(self):
    return tf.int32

  def initialize(self, name=None):
    finished = tf.tile([False], [self._batch_size])
    return (finished, self._start_inputs)

  def sample(self, time, outputs, state, name=None):
    """sample for GreedyEmbeddingHelper."""
    del time, state  # unused by sample_fn
    # Outputs are logits, use argmax to get the most probable id
    if not isinstance(outputs, tf.Tensor):
      raise TypeError("Expected outputs to be a single Tensor, got: %s" %
                      type(outputs))
    sample_ids = tf.argmax(outputs, axis=-1, output_type=tf.int32)
    return sample_ids

  def next_inputs(self, time, outputs, state, sample_ids, name=None):
    """next_inputs_fn for GreedyEmbeddingHelper."""
    del time, outputs  # unused by next_inputs_fn
    finished = tf.equal(sample_ids, self._end_token)
    all_finished = tf.reduce_all(finished)
    next_inputs = tf.cond(
        all_finished,
        # If we're finished, the next_inputs value doesn't matter
        lambda: self._start_inputs,
        lambda: tf.concat([self._embedding_fn(sample_ids), self.context], axis=-1))
    return (finished, next_inputs, state)


class ContextSampleEmbeddingHelper(ContextGreedyEmbeddingHelper):
  """A helper for use during inference.
  Uses sampling (from a distribution) instead of argmax and passes the
  result through an embedding layer to get the next input.
  """

  def __init__(self, embedding, context, start_tokens, end_token, 
               softmax_temperature=None, seed=None):
    """Initializer.
    Args:
      embedding: A callable that takes a vector tensor of `ids` (argmax ids),
        or the `params` argument for `embedding_lookup`. The returned tensor
        will be passed to the decoder input.
      start_tokens: `int32` vector shaped `[batch_size]`, the start tokens.
      end_token: `int32` scalar, the token that marks end of decoding.
      softmax_temperature: (Optional) `float32` scalar, value to divide the
        logits by before computing the softmax. Larger values (above 1.0) result
        in more random samples, while smaller values push the sampling
        distribution towards the argmax. Must be strictly greater than 0.
        Defaults to 1.0.
      seed: (Optional) The sampling seed.
    Raises:
      ValueError: if `start_tokens` is not a 1D tensor or `end_token` is not a
        scalar.
    """
    super(ContextSampleEmbeddingHelper, self).__init__(
        embedding, context, start_tokens, end_token)
    self._softmax_temperature = softmax_temperature
    self._seed = seed
    self.context = context

  def sample(self, time, outputs, state, name=None):
    """sample for SampleEmbeddingHelper."""
    del time, state  # unused by sample_fn
    # Outputs are logits, we sample instead of argmax (greedy).
    if not isinstance(outputs, tf.Tensor):
      raise TypeError("Expected outputs to be a single Tensor, got: %s" %
                      type(outputs))
    if self._softmax_temperature is None:
      logits = outputs
    else:
      logits = outputs / self._softmax_temperature
    
    sample_id_sampler = tf.distributions.Categorical(logits=logits)
    sample_ids = sample_id_sampler.sample(seed=self._seed)
    #p = tf.print(sample_ids)
    p2 = tf.print(sample_ids.shape)
    return sample_ids



# Context helpers for GPT2 sample spamGAN
class GPT2ContextGreedyEmbeddingHelper(Helper):
    """A helper for use during inference.

    Uses the argmax of the output (treated as logits) and passes the
    result through an embedding layer to get the next input.

    Note that for greedy decoding, Texar's decoders provide a simpler
    interface by specifying `decoding_strategy='infer_greedy'` when calling a
    decoder (see, e.g.,,
    :meth:`RNN decoder <texar.tf.modules.RNNDecoderBase._build>`). In this case,
    use of GreedyEmbeddingHelper is not necessary.
    """

    def __init__(self, embedding, mode, context, start_tokens, end_token):
        """Initializer.

        Args:
          embedding: A callable or the `params` argument for `embedding_lookup`.
            If a callable, it can take a vector tensor of `ids` (argmax ids),
            or take two arguments (`ids`, `times`), where `ids` is a vector
            tensor of argmax ids, and `times` is a vector tensor of current
            time steps (i.e., position ids). The latter case can be used when
            attr:`embedding` is a combination of word embedding and position
            embedding.
            The returned tensor will be returned by :meth:`next_inputs`.
          start_tokens: `int32` vector shaped `[batch_size]`, the start tokens.
          end_token: `int32` scalar, the token that marks end of decoding.

        Raises:
          ValueError: if `start_tokens` is not a 1D tensor or `end_token` is not a
            scalar.
        """
        if callable(embedding):
            self._embedding_fn = embedding
        else:
            self._embedding_fn = (
                lambda ids: tf.nn.embedding_lookup(embedding, ids))
        
        self.context = context
        self.mode = mode
        self._start_tokens = tf.convert_to_tensor(
            start_tokens, dtype=tf.int32, name="start_tokens")
        self._end_token = tf.convert_to_tensor(
            end_token, dtype=tf.int32, name="end_token")
        if self._start_tokens.get_shape().ndims != 1:
            raise ValueError("start_tokens must be a vector")
        self._batch_size = tf.size(start_tokens)
#         print("self._batch_size: {}".format(self._batch_size))
        if self._end_token.get_shape().ndims != 0:
            raise ValueError("end_token must be a scalar")
        
        self._embedding_args_cnt = len(get_args(self._embedding_fn))
        if self._embedding_args_cnt == 2:
            # Position index is 0 in the beginning
            times = tf.zeros([self._batch_size], dtype=tf.int32)
            self._start_inputs = self._embedding_fn(self._start_tokens, times)
#             print("times: {}".format(times))
#             print("self._start_inputs: {}".format(self._start_inputs))
        elif self._embedding_args_cnt == 3:
            # Position index is 0 in the beginning
            times = tf.zeros([self._batch_size], dtype=tf.int32)
            self._start_inputs = self._embedding_fn(self._start_tokens, times, self.mode)
#             print("times: {}".format(times))
#             print("self._start_inputs: {}".format(self._start_inputs))
        else:
            raise ValueError('`embedding` should expect 2 or 3 arguments.')
        
        self._start_inputs = tf.concat(
            [self._start_inputs[:, :(self._start_inputs.shape[-1]-self.context.shape[-1])], self.context], 
            axis=-1
            )

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def sample_ids_shape(self):
        return tf.TensorShape([])

    @property
    def sample_ids_dtype(self):
        return tf.int32

    def initialize(self, name=None):
        finished = tf.tile([False], [self._batch_size])
        return finished, self._start_inputs

    def sample(self, time, outputs, state, name=None):
        """Gets a sample for one step."""
        del time, state  # unused by sample_fn
        # Outputs are logits, use argmax to get the most probable id
        if not isinstance(outputs, tf.Tensor):
            raise TypeError("Expected outputs to be a single Tensor, got: %s" %
                            type(outputs))
        sample_ids = tf.argmax(outputs, axis=-1, output_type=tf.int32)
        return sample_ids

    def next_inputs(self, time, outputs, state, sample_ids, name=None):
        """Gets the inputs for next step."""
        finished = tf.equal(sample_ids, self._end_token)
        all_finished = tf.reduce_all(finished)

        if self._embedding_args_cnt == 2:
            del outputs
            # Prepare the position embedding of the next step
            times = tf.ones(self._batch_size, dtype=tf.int32) * (time + 1)
            next_inputs = tf.cond(
                all_finished,
                # If we're finished, the next_inputs value doesn't matter
                lambda: self._start_inputs,
                lambda: tf.concat(
                    [self._embedding_fn(sample_ids, times)[:, :(self._start_inputs.shape[-1]-self.context.shape[-1])], self.context], axis=-1)
                )
#             print("time {}, next_inputs: {}".format(time, next_inputs))
        elif self._embedding_args_cnt == 3:
            del outputs
            # Prepare the position embedding of the next step
            times = tf.ones(self._batch_size, dtype=tf.int32) * (time + 1)
            next_inputs = tf.cond(
                all_finished,
                # If we're finished, the next_inputs value doesn't matter
                lambda: self._start_inputs,
                lambda: tf.concat(
                    [self._embedding_fn(sample_ids, times, self.mode)[:, :(self._start_inputs.shape[-1]-self.context.shape[-1])], self.context], axis=-1)
                )
#             print("time {}, next_inputs: {}".format(time, next_inputs))
        return finished, next_inputs, state

class GPT2ContextSampleEmbeddingHelper(GPT2ContextGreedyEmbeddingHelper):
  """A helper for use during inference.
  Uses sampling (from a distribution) instead of argmax and passes the
  result through an embedding layer to get the next input.
  """

  def __init__(self, embedding, mode, context, start_tokens, end_token,
               softmax_temperature=None, seed=None):
    """Initializer.
    Args:
      embedding: A callable that takes a vector tensor of `ids` (argmax ids),
        or the `params` argument for `embedding_lookup`. The returned tensor
        will be passed to the decoder input.
      start_tokens: `int32` vector shaped `[batch_size]`, the start tokens.
      end_token: `int32` scalar, the token that marks end of decoding.
      softmax_temperature: (Optional) `float32` scalar, value to divide the
        logits by before computing the softmax. Larger values (above 1.0) result
        in more random samples, while smaller values push the sampling
        distribution towards the argmax. Must be strictly greater than 0.
        Defaults to 1.0.
      seed: (Optional) The sampling seed.
    Raises:
      ValueError: if `start_tokens` is not a 1D tensor or `end_token` is not a
        scalar.
    """
    super(GPT2ContextSampleEmbeddingHelper, self).__init__(
        embedding, mode, context, start_tokens, end_token)
    self._softmax_temperature = softmax_temperature
    self._seed = seed
    self.context = context
    self.mode = mode

  def sample(self, time, outputs, state, name=None):
    """sample for SampleEmbeddingHelper."""
    del time, state  # unused by sample_fn
    # Outputs are logits, we sample instead of argmax (greedy).
    if not isinstance(outputs, tf.Tensor):
      raise TypeError("Expected outputs to be a single Tensor, got: %s" %
                      type(outputs))
    if self._softmax_temperature is None:
      logits = outputs
    else:
      logits = outputs / self._softmax_temperature
    
    sample_id_sampler = tf.distributions.Categorical(logits=logits)
    sample_ids = sample_id_sampler.sample(seed=self._seed)
    #p = tf.print(sample_ids)
    p2 = tf.print(sample_ids.shape)
    return sample_ids


def _top_k_logits(logits, k):
    """Adapted from
    https://github.com/openai/gpt-2/blob/master/src/sample.py#L63-L77
    """
    if k == 0:
        # no truncation
        return logits

    def _top_k():
        values, _ = tf.nn.top_k(logits, k=k)
        min_values = values[:, -1, tf.newaxis]
        return tf.where(
            logits < min_values,
            tf.ones_like(logits, dtype=logits.dtype) * -1e10,
            logits,
        )
    return tf.cond(
        tf.equal(k, 0),
        lambda: logits,
        lambda: _top_k(),
    )


class GPT2ContextTopKSampleEmbeddingHelper(GPT2ContextGreedyEmbeddingHelper):
    """A helper for use during inference.

    Samples from `top_k` most likely candidates from a vocab distribution,
    and passes the result through an embedding layer to get the next input.
    """

    def __init__(self, embedding, mode, context, start_tokens, end_token, top_k=10,
                 softmax_temperature=None, seed=None):
        """Initializer.

        Args:
            embedding: A callable or the `params` argument for
                `embedding_lookup`. If a callable, it can take a vector tensor
                of token `ids`, or take two arguments (`ids`, `times`),
                where `ids` is a vector
                tensor of token ids, and `times` is a vector tensor of current
                time steps (i.e., position ids). The latter case can be used
                when attr:`embedding` is a combination of word embedding and
                position embedding.
            start_tokens: `int32` vector shaped `[batch_size]`, the start
                tokens.
            end_token: `int32` scalar, the token that marks end of decoding.
            top_k: `int32` scalar tensor. Number of top candidates to sample
                from. Must be `>=0`. If set to 0, samples from all candidates
                (i.e., regular random sample decoding).
            softmax_temperature (optional): `float32` scalar, value to
                divide the logits by before computing the softmax. Larger values
                (above 1.0) result in more random samples, while smaller values
                push the sampling distribution towards the argmax. Must be
                strictly greater than 0. Defaults to 1.0.
            seed (optional): The sampling seed.

        Raises:
            ValueError: if `start_tokens` is not a 1D tensor or `end_token` is
            not a scalar.
        """
        super(GPT2ContextTopKSampleEmbeddingHelper, self).__init__(
            embedding, mode, context, start_tokens, end_token)
        self._top_k = top_k
        self._softmax_temperature = softmax_temperature
        self._seed = seed
        self.context = context
        self.mode = mode

    def sample(self, time, outputs, state, name=None):
        """Gets a sample for one step."""
        del time, state  # unused by sample_fn
        # Outputs are logits, we sample from the top_k candidates
        if not isinstance(outputs, tf.Tensor):
            raise TypeError("Expected outputs to be a single Tensor, got: %s" %
                            type(outputs))
        if self._softmax_temperature is None:
            logits = outputs
        else:
            logits = outputs / self._softmax_temperature

        logits = _top_k_logits(logits, k=self._top_k)

        sample_id_sampler = tfpd.Categorical(logits=logits)
        sample_ids = sample_id_sampler.sample(seed=self._seed)

        return sample_ids


class GPT2ContextSoftmaxEmbeddingHelper(Helper):
    """A helper that feeds softmax probabilities over vocabulary
    to the next step.
    Uses the softmax probability vector to pass through word embeddings to
    get the next input (i.e., a mixed word embedding).

    A subclass of
    :tf_main:`Helper <contrib/seq2seq/Helper>`.
    Used as a helper to :class:`~texar.tf.modules.RNNDecoderBase` :meth:`_build`
    in inference mode.

    Args:
        embedding: A callable or the `params` argument for
            :tf_main:`tf.nn.embedding_lookup <nn/embedding_lookup>`.
            If a callable, it can take a float tensor named `soft_ids` which is
            a distribution over indexes. For example, the shape of the tensor
            is typically `[batch_size, vocab_size]`. The callable can also
            take two arguments (`soft_ids`, `times`), where `soft_ids` is
            as above, and `times` is an int vector tensor of current
            time steps (i.e., position ids). The latter case can be used
            when attr:`embedding` is a combination of word embedding and
            position embedding.
        start_tokens: An int tensor shaped `[batch_size]`. The
            start tokens.
        end_token: An int scalar tensor. The token that marks end of
            decoding.
        tau: A float scalar tensor, the softmax temperature.
        embedding_size (optional): An int scalar tensor, the number of
            embedding vectors. Usually it is the vocab size. Required if
            :attr:`embedding` is a callable.
        stop_gradient (bool): Whether to stop the gradient backpropagation
            when feeding softmax vector to the next step.
        use_finish (bool): Whether to stop decoding once `end_token` is
            generated. If `False`, decoding will continue until
            `max_decoding_length` of the decoder is reached.
    """

    def __init__(self, embedding, mode, context, start_tokens, end_token, tau,
                 embedding_size=None, stop_gradient=False, use_finish=True):
        if callable(embedding):
            self._embedding_fn = embedding

            if embedding_size is None:
                raise ValueError('`embedding_size` must be provided if '
                                 '`embedding` is a callable.')
            self._embedding_size = tf.convert_to_tensor(
                embedding_size, dtype=tf.int32, name="embedding_size")
        else:
            self._embedding_fn = (
                lambda soft_ids: soft_embedding_lookup(embedding, soft_ids))
            self._embedding_size = tf.shape(embedding)[0]
            
        self.context = context
        self.mode = mode
        self._start_tokens = tf.convert_to_tensor(
            start_tokens, dtype=tf.int32, name="start_tokens")
        self._end_token = tf.convert_to_tensor(
            end_token, dtype=tf.int32, name="end_token")
        if self._start_tokens.get_shape().ndims != 1:
            raise ValueError("start_tokens must be a vector")
        self._batch_size = array_ops.size(start_tokens)
        if self._end_token.get_shape().ndims != 0:
            raise ValueError("end_token must be a scalar")

        soft_start_tokens = tf.one_hot(
            self._start_tokens, self._embedding_size, dtype=tf.float32)
        self._embedding_args_cnt = len(utils.get_args(self._embedding_fn))
        if self._embedding_args_cnt == 2:
            # Position index is 0 in the beginning
            times = tf.zeros([self._batch_size], dtype=tf.int32)
            self._start_inputs = self._embedding_fn(
                soft_start_tokens, times)
        elif self._embedding_args_cnt == 3:
            # Position index is 0 in the beginning
            times = tf.zeros([self._batch_size], dtype=tf.int32)
            self._start_inputs = self._embedding_fn(
                soft_start_tokens, times, self.mode)
        else:
            raise ValueError('`embedding` should expect 2 or 3 arguments.')
        
        self._start_inputs = tf.concat(
            [self._start_inputs[:, :(self._start_inputs.shape[-1]-self.context.shape[-1])], self.context], 
            axis=-1
            )

        self._batch_size = tf.size(self._start_tokens)
        self._tau = tau
        self._stop_gradient = stop_gradient
        self._use_finish = use_finish

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def sample_ids_dtype(self):
        return tf.float32

    @property
    def sample_ids_shape(self):
        # A trick to convert a scalar Tensor `self._embedding_size` to
        # a `TensorShape`
        oh = tf.one_hot(0, self._embedding_size)
        return oh.get_shape()[:1]

    def initialize(self, name=None):
        finished = tf.tile([False], [self._batch_size])
        return (finished, self._start_inputs)

    def sample(self, time, outputs, state, name=None):
        """Returns `sample_id` which is softmax distributions over vocabulary
        with temperature `tau`. Shape = `[batch_size, vocab_size]`
        """
        sample_ids = tf.nn.softmax(outputs / self._tau)
        return sample_ids

    def next_inputs(self, time, outputs, state, sample_ids, name=None):
        if self._use_finish:
            hard_ids = tf.argmax(sample_ids, axis=-1, output_type=tf.int32)
            finished = tf.equal(hard_ids, self._end_token)
        else:
            finished = tf.tile([False], [self._batch_size])
        all_finished = tf.reduce_all(finished)

        if self._stop_gradient:
            sample_ids = tf.stop_gradient(sample_ids)

        if self._embedding_args_cnt == 2:
            # Prepare the position embedding of the next step
            times = tf.ones(self._batch_size, dtype=tf.int32) * (time + 1)
            next_inputs = tf.cond(
                all_finished,
                # If we're finished, the next_inputs value doesn't matter
                lambda: self._start_inputs,
                lambda: tf.concat(
                    [self._embedding_fn(sample_ids, times)[:, :(self._start_inputs.shape[-1]-self.context.shape[-1])], self.context], axis=-1)
                )
        elif self._embedding_args_cnt == 3:
            # Prepare the position embedding of the next step
            times = tf.ones(self._batch_size, dtype=tf.int32) * (time + 1)
            next_inputs = tf.cond(
                all_finished,
                # If we're finished, the next_inputs value doesn't matter
                lambda: self._start_inputs,
                lambda: tf.concat(
                    [self._embedding_fn(sample_ids, times, self.mode)[:, :(self._start_inputs.shape[-1]-self.context.shape[-1])], self.context], axis=-1)
                )
        return (finished, next_inputs, state)


class GPT2ContextGumbelSoftmaxEmbeddingHelper(GPT2ContextSoftmaxEmbeddingHelper):
    """A helper that feeds gumbel softmax sample to the next step.
    Uses the gumbel softmax vector to pass through word embeddings to
    get the next input (i.e., a mixed word embedding).

    A subclass of
    :tf_main:`Helper <contrib/seq2seq/Helper>`.
    Used as a helper to :class:`~texar.tf.modules.RNNDecoderBase` :meth:`_build`
    in inference mode.

    Same as :class:`~texar.tf.modules.SoftmaxEmbeddingHelper` except that here
    gumbel softmax (instead of softmax) is used.

    Args:
        embedding: A callable or the `params` argument for
            :tf_main:`tf.nn.embedding_lookup <nn/embedding_lookup>`.
            If a callable, it can take a float tensor named `soft_ids` which is
            a distribution over indexes. For example, the shape of the tensor
            is typically `[batch_size, vocab_size]`. The callable can also
            take two arguments (`soft_ids`, `times`), where `soft_ids` is
            as above, and `times` is an int vector tensor of current
            time steps (i.e., position ids). The latter case can be used
            when attr:`embedding` is a combination of word embedding and
            position embedding.
        start_tokens: An int tensor shaped `[batch_size]`. The
            start tokens.
        end_token: An int scalar tensor. The token that marks end of
            decoding.
        tau: A float scalar tensor, the softmax temperature.
        embedding_size (optional): An int scalar tensor, the number of
            embedding vectors. Usually it is the vocab size. Required if
            :attr:`embedding` is a callable.
        straight_through (bool): Whether to use straight through gradient
            between time steps. If `True`, a single token with highest
            probability (i.e., greedy sample) is fed to the next step and
            gradient is computed using straight through. If `False` (default),
            the soft gumbel-softmax distribution is fed to the next step.
        stop_gradient (bool): Whether to stop the gradient backpropagation
            when feeding softmax vector to the next step.
        use_finish (bool): Whether to stop decoding once `end_token` is
            generated. If `False`, decoding will continue until
            `max_decoding_length` of the decoder is reached.
    """
    def __init__(self, embedding, mode, context, start_tokens, end_token, tau,
                 embedding_size=None, straight_through=False,
                 stop_gradient=False, use_finish=True):
        super(GPT2ContextGumbelSoftmaxEmbeddingHelper, self).__init__(
            embedding, mode, context, start_tokens, end_token, tau, embedding_size,
            stop_gradient, use_finish)
        self._straight_through = straight_through
        self.mode = mode
        self.context = context

    def sample(self, time, outputs, state, name=None):
        """Returns `sample_id` of shape `[batch_size, vocab_size]`. If
        `straight_through` is False, this is gumbel softmax distributions over
        vocabulary with temperature `tau`. If `straight_through` is True,
        this is one-hot vectors of the greedy samples.
        """
        sample_ids = tf.nn.softmax(outputs / self._tau)
        sample_ids = tfpd.RelaxedOneHotCategorical(
            self._tau, logits=outputs).sample()
        if self._straight_through:
            size = tf.shape(sample_ids)[-1]
            sample_ids_hard = tf.cast(
                tf.one_hot(tf.argmax(sample_ids, -1), size), sample_ids.dtype)
            sample_ids = tf.stop_gradient(sample_ids_hard - sample_ids) \
                         + sample_ids
        return sample_ids


class TrainingHelper(Helper):
    """A helper for use during training. Performs teacher-forcing decoding.

    Returned sample_ids are the argmax of the RNN output logits.

    Note that for teacher-forcing decoding, Texar's decoders provide a simpler
    interface by specifying `decoding_strategy='train_greedy'` when calling a
    decoder (see, e.g.,,
    :meth:`RNN decoder <texar.tf.modules.RNNDecoderBase._build>`). In this case,
    use of TrainingHelper is not necessary.
    """

    def __init__(self, inputs, sequence_length, time_major=False, name=None):
        """Initializer.

        Args:
          inputs: A (structure of) input tensors.
          sequence_length: An int32 vector tensor.
          time_major: Python bool.  Whether the tensors in `inputs` are time major.
            If `False` (default), they are assumed to be batch major.
          name: Name scope for any created operations.

        Raises:
          ValueError: if `sequence_length` is not a 1D tensor.
        """
        with ops.name_scope(name, "TrainingHelper", [inputs, sequence_length]):
            inputs = ops.convert_to_tensor(inputs, name="inputs")
            self._inputs = inputs
            if not time_major:
                inputs = nest.map_structure(_transpose_batch_time, inputs)

            self._input_tas = nest.map_structure(_unstack_ta, inputs)
            self._sequence_length = ops.convert_to_tensor(
                sequence_length, name="sequence_length")
            if self._sequence_length.get_shape().ndims != 1:
                raise ValueError(
                    "Expected sequence_length to be a vector, but received shape: %s" %
                    self._sequence_length.get_shape())

            self._zero_inputs = nest.map_structure(
                lambda inp: array_ops.zeros_like(inp[0, :]), inputs)
            self._start_inputs = self._zero_inputs
            self._batch_size = shape_list(sequence_length)[0]

    @property
    def inputs(self):
        return self._inputs

    @property
    def sequence_length(self):
        return self._sequence_length

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def sample_ids_shape(self):
        return tensor_shape.TensorShape([])

    @property
    def sample_ids_dtype(self):
        return dtypes.int32

    def initialize(self, name=None):
        with ops.name_scope(name, "TrainingHelperInitialize"):
            finished = math_ops.equal(0, self._sequence_length)
            all_finished = math_ops.reduce_all(finished)
            next_inputs = control_flow_ops.cond(
                all_finished, lambda: self._zero_inputs,
                lambda: nest.map_structure(lambda inp: inp.read(0), self._input_tas))
            return (finished, next_inputs)

    def sample(self, time, outputs, name=None, **unused_kwargs):
        """Gets a sample for one step."""
        with ops.name_scope(name, "TrainingHelperSample", [time, outputs]):
            sample_ids = math_ops.cast(
                math_ops.argmax(outputs, axis=-1), dtypes.int32)
            return sample_ids

    def next_inputs(self, time, outputs, state, name=None, **unused_kwargs):
        """Gets the inputs for next step."""
        with ops.name_scope(name, "TrainingHelperNextInputs",
                            [time, outputs, state]):
            next_time = time + 1
            finished = (next_time >= self._sequence_length)
            all_finished = math_ops.reduce_all(finished)

            def read_from_ta(inp):
                return inp.read(next_time)

            next_inputs = control_flow_ops.cond(
                all_finished, lambda: self._zero_inputs,
                lambda: nest.map_structure(read_from_ta, self._input_tas))
            return (finished, next_inputs, state)


class GPT2ScheduledEmbeddingTrainingHelper(TrainingHelper):
    """A training helper that adds scheduled sampling.

    Returns -1s for sample_ids where no sampling took place; valid sample id
    values elsewhere.
    """

    def __init__(self, inputs, sequence_length, embedding, mode, context, sampling_probability,
                 time_major=False, seed=None, scheduling_seed=None, name=None):
        """Initializer.

        Args:
          inputs: A (structure of) input tensors.
          sequence_length: An int32 vector tensor.
          embedding: A callable or the `params` argument for `embedding_lookup`.
            If a callable, it can take a vector tensor of token `ids`,
            or take two arguments (`ids`, `times`), where `ids` is a vector
            tensor of token ids, and `times` is a vector tensor of current
            time steps (i.e., position ids). The latter case can be used when
            attr:`embedding` is a combination of word embedding and position
            embedding.
          sampling_probability: A 0D `float32` tensor: the probability of sampling
            categorically from the output ids instead of reading directly from the
            inputs.
          time_major: Python bool.  Whether the tensors in `inputs` are time major.
            If `False` (default), they are assumed to be batch major.
          seed: The sampling seed.
          scheduling_seed: The schedule decision rule sampling seed.
          name: Name scope for any created operations.

        Raises:
          ValueError: if `sampling_probability` is not a scalar or vector.
        """
        with ops.name_scope(name, "GPT2ScheduledEmbeddingTrainingHelper",
                            [embedding, sampling_probability]):
            if callable(embedding):
                self._embedding_fn = embedding
            else:
                self._embedding_fn = (
                    lambda ids: embedding_ops.embedding_lookup(embedding, ids))

            self._embedding_args_cnt = len(get_args(self._embedding_fn))
            if self._embedding_args_cnt != 2 and self._embedding_args_cnt != 3:
                raise ValueError('`embedding` should expect 2 or 3 arguments.')

            self._sampling_probability = ops.convert_to_tensor(
                sampling_probability, name="sampling_probability")
            if self._sampling_probability.get_shape().ndims not in (0, 1):
                raise ValueError(
                    "sampling_probability must be either a scalar or a vector. "
                    "saw shape: %s" % (self._sampling_probability.get_shape()))
            self._seed = seed
            self._scheduling_seed = scheduling_seed
            self.mode = mode
            self.context = context
            super(GPT2ScheduledEmbeddingTrainingHelper, self).__init__(
                inputs=inputs,
                sequence_length=sequence_length,
                time_major=time_major,
                name=name)

    def initialize(self, name=None):
        return super(GPT2ScheduledEmbeddingTrainingHelper, self).initialize(
            name=name)

    def sample(self, time, outputs, state, name=None):
        """Gets a sample for one step."""
        with ops.name_scope(name, "GPT2ScheduledEmbeddingTrainingHelperSample",
                            [time, outputs, state]):
            # Return -1s where we did not sample, and sample_ids elsewhere
            select_sampler = tfpd.Bernoulli(
                probs=self._sampling_probability, dtype=dtypes.bool)
            select_sample = select_sampler.sample(
                sample_shape=self.batch_size, seed=self._scheduling_seed)
            sample_id_sampler = tfpd.Categorical(logits=outputs)
            return array_ops.where(
                select_sample,
                sample_id_sampler.sample(seed=self._seed),
                gen_array_ops.fill([self.batch_size], -1))

    def next_inputs(self, time, outputs, state, sample_ids, name=None):
        """Gets the outputs for next step."""
        with ops.name_scope(name, "GPT2ScheduledEmbeddingTrainingHelperNextInputs",
                            [time, outputs, state, sample_ids]):
            (finished, base_next_inputs, state) = (
                super(GPT2ScheduledEmbeddingTrainingHelper, self).next_inputs(
                    time=time,
                    outputs=outputs,
                    state=state,
                    sample_ids=sample_ids,
                    name=name))

            def maybe_sample():
                """Perform scheduled sampling."""
                where_sampling = math_ops.cast(
                    array_ops.where(sample_ids > -1), dtypes.int32)
                where_not_sampling = math_ops.cast(
                    array_ops.where(sample_ids <= -1), dtypes.int32)
                sample_ids_sampling = array_ops.gather_nd(sample_ids, where_sampling)
                inputs_not_sampling = array_ops.gather_nd(
                    base_next_inputs, where_not_sampling)

                if self._embedding_args_cnt == 2:
                    # Prepare the position embedding of the next step
                    times = tf.ones(self._batch_size,
                                    dtype=tf.int32) * (time + 1)
                    sampled_next_inputs = self._embedding_fn(sample_ids_sampling, times)
                    sampled_next_inputs = tf.concat(
                        [sampled_next_inputs[:, :(sampled_next_inputs.shape[-1]-self.context.shape[-1])], self.context], axis=-1)
                elif self._embedding_args_cnt == 3:
                    # Prepare the position embedding of the next step
                    times = tf.ones(self._batch_size,
                                    dtype=tf.int32) * (time + 1)
                    sampled_next_inputs = self._embedding_fn(sample_ids_sampling, times, self.mode)
#                     sampled_next_inputs = tf.concat(
#                         [sampled_next_inputs[:, :(sampled_next_inputs.shape[-1]-self.context.shape[-1])], self.context], axis=-1)
                base_shape = array_ops.shape(base_next_inputs)
                return (array_ops.scatter_nd(indices=where_sampling,
                                             updates=sampled_next_inputs,
                                             shape=base_shape)
                        + array_ops.scatter_nd(indices=where_not_sampling,
                                               updates=inputs_not_sampling,
                                               shape=base_shape))

            all_finished = math_ops.reduce_all(finished)
            next_inputs = control_flow_ops.cond(
                all_finished, lambda: base_next_inputs, maybe_sample)
            return (finished, next_inputs, state)


class GPT2ScheduledOutputTrainingHelper(TrainingHelper):
    """A training helper that adds scheduled sampling directly to outputs.

    Returns False for sample_ids where no sampling took place; True elsewhere.
    """

    def __init__(self, inputs, sequence_length, sampling_probability,
                 time_major=False, seed=None, next_inputs_fn=None,
                 auxiliary_inputs=None, name=None):
        """Initializer.

        Args:
          inputs: A (structure) of input tensors.
          sequence_length: An int32 vector tensor.
          sampling_probability: A 0D `float32` tensor: the probability of sampling
            from the outputs instead of reading directly from the inputs.
          time_major: Python bool.  Whether the tensors in `inputs` are time major.
            If `False` (default), they are assumed to be batch major.
          seed: The sampling seed.
          next_inputs_fn: (Optional) callable to apply to the RNN outputs to create
            the next input when sampling. If `None` (default), the RNN outputs will
            be used as the next inputs.
          auxiliary_inputs: An optional (structure of) auxiliary input tensors with
            a shape that matches `inputs` in all but (potentially) the final
            dimension. These tensors will be concatenated to the sampled output or
            the `inputs` when not sampling for use as the next input.
          name: Name scope for any created operations.

        Raises:
          ValueError: if `sampling_probability` is not a scalar or vector.
        """
        with ops.name_scope(name, "GPT2ScheduledOutputTrainingHelper",
                            [inputs, auxiliary_inputs, sampling_probability]):
            self._sampling_probability = ops.convert_to_tensor(
                sampling_probability, name="sampling_probability")
            if self._sampling_probability.get_shape().ndims not in (0, 1):
                raise ValueError(
                    "sampling_probability must be either a scalar or a vector. "
                    "saw shape: %s" % (self._sampling_probability.get_shape()))

            if auxiliary_inputs is None:
                maybe_concatenated_inputs = inputs
            else:
                inputs = ops.convert_to_tensor(inputs, name="inputs")
                auxiliary_inputs = ops.convert_to_tensor(
                    auxiliary_inputs, name="auxiliary_inputs")
                maybe_concatenated_inputs = nest.map_structure(
                    lambda x, y: array_ops.concat((x, y), -1),
                    inputs, auxiliary_inputs)
                if not time_major:
                    auxiliary_inputs = nest.map_structure(
                        _transpose_batch_time, auxiliary_inputs)

            self._auxiliary_input_tas = (
                nest.map_structure(_unstack_ta, auxiliary_inputs)
                if auxiliary_inputs is not None else None)

            self._seed = seed

            self._next_inputs_fn = next_inputs_fn

            super(GPT2ScheduledOutputTrainingHelper, self).__init__(
                inputs=maybe_concatenated_inputs,
                sequence_length=sequence_length,
                time_major=time_major,
                name=name)

    def initialize(self, name=None):
        return super(GPT2ScheduledOutputTrainingHelper, self).initialize(name=name)

    def sample(self, time, outputs, state, name=None):
        """Gets a sample for one step."""
        with ops.name_scope(name, "ScheduledOutputTrainingHelperSample",
                            [time, outputs, state]):
            sampler = tfpd.Bernoulli(probs=self._sampling_probability)
            return sampler.sample(sample_shape=self.batch_size, seed=self._seed)

    def next_inputs(self, time, outputs, state, sample_ids, name=None):
        """Gets the next inputs for next step."""
        with ops.name_scope(name, "ScheduledOutputTrainingHelperNextInputs",
                            [time, outputs, state, sample_ids]):
            (finished, base_next_inputs, state) = (
                super(GPT2ScheduledOutputTrainingHelper, self).next_inputs(
                    time=time,
                    outputs=outputs,
                    state=state,
                    sample_ids=sample_ids,
                    name=name))
            sample_ids = math_ops.cast(sample_ids, dtypes.bool)

            def maybe_sample():
                """Perform scheduled sampling."""

                def maybe_concatenate_auxiliary_inputs(outputs_, indices=None):
                    """Concatenate outputs with auxiliary inputs, if they exist."""
                    if self._auxiliary_input_tas is None:
                        return outputs_

                    next_time = time + 1
                    auxiliary_inputs = nest.map_structure(
                        lambda ta: ta.read(next_time), self._auxiliary_input_tas)
                    if indices is not None:
                        auxiliary_inputs = array_ops.gather_nd(auxiliary_inputs, indices)
                    return nest.map_structure(
                        lambda x, y: array_ops.concat((x, y), -1),
                        outputs_, auxiliary_inputs)

                if self._next_inputs_fn is None:
                    return array_ops.where(
                        sample_ids, maybe_concatenate_auxiliary_inputs(outputs),
                        base_next_inputs)

                where_sampling = math_ops.cast(
                    array_ops.where(sample_ids), dtypes.int32)
                where_not_sampling = math_ops.cast(
                    array_ops.where(math_ops.logical_not(sample_ids)), dtypes.int32)
                outputs_sampling = array_ops.gather_nd(outputs, where_sampling)
                inputs_not_sampling = array_ops.gather_nd(base_next_inputs,
                                                          where_not_sampling)
                sampled_next_inputs = maybe_concatenate_auxiliary_inputs(
                    self._next_inputs_fn(outputs_sampling), where_sampling)

                base_shape = array_ops.shape(base_next_inputs)
                return (array_ops.scatter_nd(indices=where_sampling,
                                             updates=sampled_next_inputs,
                                             shape=base_shape)
                        + array_ops.scatter_nd(indices=where_not_sampling,
                                               updates=inputs_not_sampling,
                                               shape=base_shape))

            all_finished = math_ops.reduce_all(finished)
            no_samples = math_ops.logical_not(math_ops.reduce_any(sample_ids))
            next_inputs = control_flow_ops.cond(
                math_ops.logical_or(all_finished, no_samples),
                lambda: base_next_inputs, maybe_sample)
            return (finished, next_inputs, state)