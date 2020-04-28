from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import texar.tf as tx
from texar.tf.utils.utils import get_args

from tensorflow.python.ops import array_ops
from tensorflow_probability import distributions as tfpd

from texar.tf.modules.embedders.embedder_utils import soft_embedding_lookup
from texar.tf.utils import utils



## MODIFIED TO ALLOW CLASS EMBEDDING APPENDING
class ContextSoftmaxEmbeddingHelper(tf.contrib.seq2seq.Helper):
    """A helper that feeds softmax probabilities over vocabulary
    to the next step.
    Uses the softmax probability vector to pass through word embeddings to
    get the next input (i.e., a mixed word embedding).
    A subclass of
    :tf_main:`Helper <contrib/seq2seq/Helper>`.
    Used as a helper to :class:`~texar.modules.RNNDecoderBase` :meth:`_build`
    in inference mode.
    Args:
        embedding: An embedding argument (:attr:`params`) for
            :tf_main:`tf.nn.embedding_lookup <nn/embedding_lookup>`, or an
            instance of subclass of :class:`texar.modules.EmbedderBase`.
            Note that other callables are not acceptable here.
        start_tokens: An int tensor shaped `[batch_size]`. The
            start tokens.
        end_token: An int scalar tensor. The token that marks end of
            decoding.
        tau: A float scalar tensor, the softmax temperature.
        stop_gradient (bool): Whether to stop the gradient backpropagation
            when feeding softmax vector to the next step.
        use_finish (bool): Whether to stop decoding once `end_token` is
            generated. If `False`, decoding will continue until
            `max_decoding_length` of the decoder is reached.
    """

    def __init__(self, embedding, context, start_tokens, end_token, tau,
                 stop_gradient=False, use_finish=True):
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
        self._start_inputs = self._embedding_fn(self._start_tokens)
        self._batch_size = tf.size(self._start_tokens)
        self._start_inputs = tf.concat([self._start_inputs, self.context], axis=-1)
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
        return self._embedding.get_shape()[:1]

    def initialize(self, name=None):
        finished = tf.tile([False], [self._batch_size])
        return (finished, self._start_inputs)

    def sample(self, time, outputs, state, name=None):
        """Returns `sample_id` which is softmax distributions over vocabulary
        with temperature `tau`. Shape = `[batch_size, vocab_size]`
        """
        sample_dist = tf.nn.softmax(outputs / self._tau)
        sampler = tf.distributions.Categorical(logits=sample_dist)
        sample_ids = sampler.sample()
        return sample_ids

    def next_inputs(self, time, outputs, state, sample_ids, name=None):
        if self._use_finish:
            hard_ids = tf.argmax(sample_ids, axis=-1, output_type=tf.int32)
            finished = tf.equal(hard_ids, self._end_token)
        else:
            finished = tf.tile([False], [self._batch_size])
        if self._stop_gradient:
            sample_ids = tf.stop_gradient(sample_ids)
        next_inputs = self._embedding_fn(sample_ids)
        ## Modified
        next_inputs = tf.concat([next_inputs, self.context], axis=-1)
        return (finished, next_inputs, state)


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


