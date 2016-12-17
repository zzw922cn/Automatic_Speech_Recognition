# -*- coding:utf-8 -*-
''' decoder for sequence-to-sequence learning for lyrics generation or
    automatic speech recognition, revised from tensorflow source code,
    this code will be updated later to include some other features.
author:

      iiiiiiiiiiii            iiiiiiiiiiii         !!!!!!!             !!!!!!    
      #        ###            #        ###           ###        I#        #:     
      #      ###              #      I##;             ##;       ##       ##      
            ###                     ###               !##      ####      #       
           ###                     ###                 ###    ## ###    #'       
         !##;                    `##%                   ##;  ##   ###  ##        
        ###                     ###                     $## `#     ##  #         
       ###        #            ###        #              ####      ####;         
     `###        -#           ###        `#               ###       ###          
     ##############          ##############               `#         #     
     
date:2016-12-07
'''


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange  
from six.moves import zip    

import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import variable_scope
from tensorflow.python.util import nest

linear = rnn_cell._linear  # pylint: disable=protected-access


def _extract_argmax_and_embed(embedding, output_projection=None,
                              update_embedding=True):
  """这个函数的作用实际上就是定义了一个loop_function
  Get a loop_function that extracts the previous symbol and embeds it.

  Args:
    embedding: embedding tensor for symbols.
    output_projection: None or a pair (W, B). If provided, each fed previous
      output will first be multiplied by W and added B.
    update_embedding: Boolean; if False, the gradients will not propagate
      through the embeddings.

  Returns:
    A loop function.
    #这里可以确定是否对embedding矩阵进行更新，一般情况下我们选择更新
  """
  def loop_function(prev, _):
    if output_projection is not None:
      prev = nn_ops.xw_plus_b(
          prev, output_projection[0], output_projection[1])
    prev_symbol = math_ops.argmax(prev, 1)
    emb_prev = embedding_ops.embedding_lookup(embedding, prev_symbol)
    if not update_embedding:
      emb_prev = array_ops.stop_gradient(emb_prev)
    return emb_prev
  return loop_function


def rnn_decoder(decoder_inputs, initial_state, cell, loop_function=None,
                scope=None):
  """RNN decoder for the sequence-to-sequence model.

  Args:
    decoder_inputs: 解码器的输入，是一个列表，其单个元素维度是[batch_size, input_size].
    initial_state: 解码器的隐含状态，是一个列表，元素维度为[batch_size, cell.state_size].
    也就是说，以上两个变量对时间长度都是没有要求的
    
    cell: 必须是符合rnn_cell.RNNCell类，定义了函数以及大小，这里我们可以参考conv_lstm的代码实现
    loop_function: If not None, this function will be applied to the i-th output
      in order to generate the i+1-st input, and decoder_inputs will be ignored,
      except for the first element ("GO" symbol). This can be used for decoding,
      but also for training to emulate http://arxiv.org/abs/1506.03099.
      Signature -- loop_function(prev, i) = next
        * prev is a 2D Tensor of shape [batch_size x output_size],
        * i is an integer, the step number (when advanced control is needed),
        * next is a 2D Tensor of shape [batch_size x input_size].
    scope: VariableScope for the created subgraph; defaults to "rnn_decoder".

  Returns:
    A tuple of the form (outputs, state), where:
      outputs: 解码器的输出，维度同decoder_inputs一样，可以看作分别是RNN的输出和输入
      state: 最后每一时刻解码器的所有隐含状态，维度同initial_state一样
	值得注意的是对于RNN和GRU而言，output与state是一样的，但是对于LSTM而言不是
  """
  with variable_scope.variable_scope(scope or "rnn_decoder"):
    state = initial_state
    outputs = []
    prev = None
    for i, inp in enumerate(decoder_inputs):
      #实际上是按照时间顺序循环
      if loop_function is not None and prev is not None:
	#这个时候实际上decoder_inputs已经不起作用了
        with variable_scope.variable_scope("loop_function", reuse=True):
          inp = loop_function(prev, i)
      if i > 0:
	#这个意思是t>1时刻的参数与之前的参数共享
        variable_scope.get_variable_scope().reuse_variables()
      #将inp以及state传入RNNcell中，得到output以及state
      output, state = cell(inp, state)
      outputs.append(output)
      if loop_function is not None:
        prev = output
  return outputs, state


def basic_rnn_seq2seq(
    encoder_inputs, decoder_inputs, cell, dtype=dtypes.float32, scope=None):
  """一个最基本完整的seq2seq模型
	encoder与decoder使用同样类型的cell，但是不共享参数
    参数只比rnn_decoder多了一个encoder_inputs
    也就是说如果我们不希望编码器和解码器采取同样的cell，那么
    我们就自己写encoder，然后将其最后状态传入rnn_decoder

  Args:
    encoder_inputs: A list of 2D Tensors [batch_size x input_size].
    decoder_inputs: A list of 2D Tensors [batch_size x input_size].
    cell: rnn_cell.RNNCell defining the cell function and size.
    dtype: The dtype of the initial state of the RNN cell (default: tf.float32).
    scope: VariableScope for the created subgraph; default: "basic_rnn_seq2seq".
  Returns:
    A tuple of the form (outputs, state), where:
      outputs: A list of the same length as decoder_inputs of 2D Tensors with
        shape [batch_size x output_size] containing the generated outputs.
      state: The state of each decoder cell in the final time-step.
        It is a 2D Tensor of shape [batch_size x cell.state_size].
  """
  with variable_scope.variable_scope(scope or "basic_rnn_seq2seq"):
    #这里其实很简单，即先把encoder_inputs传入到rnn，然后即可传入到rnn_decoder
    #这里的enc-state是最后时刻的状态
    #所以如果我们要构造一个比较新颖的编码器，第一步就是要构造一个心音的
    #rnncell，然后将其传入rnn.rnn即可得到最终的状态输出，
    #然后将该状态作为输入传入到解码器
    _, enc_state = rnn.rnn(cell, encoder_inputs, dtype=dtype)
    return rnn_decoder(decoder_inputs, enc_state, cell)


def embedding_rnn_decoder(decoder_inputs,
                          initial_state,
                          cell,
                          num_symbols,
                          embedding_size,
                          output_projection=None,
                          feed_previous=False,
                          update_embedding_for_previous=True,
                          scope=None):
  """自带词嵌入的rnn解码器

  Args:
    decoder_inputs: A list of 1D batch-sized int32 Tensors (decoder inputs).
    initial_state: 2D Tensor [batch_size x cell.state_size].
    cell: rnn_cell.RNNCell defining the cell function.
    num_symbols: Integer, how many symbols come into the embedding.
    embedding_size: Integer, the length of the embedding vector for each symbol.
    output_projection: None or a pair (W, B) of output projection weights and
      biases; W has shape [output_size x num_symbols] and B has
      shape [num_symbols]; if provided and feed_previous=True, each fed
      previous output will first be multiplied by W and added B.
    feed_previous: Boolean; if True, only the first of decoder_inputs will be
      used (the "GO" symbol), and all other decoder inputs will be generated by:
        next = embedding_lookup(embedding, argmax(previous_output)),
      In effect, this implements a greedy decoder. It can also be used
      during training to emulate http://arxiv.org/abs/1506.03099.
      If False, decoder_inputs are used as given (the standard decoder case).
    update_embedding_for_previous: Boolean; if False and feed_previous=True,
      only the embedding for the first symbol of decoder_inputs (the "GO"
      symbol) will be updated by back propagation. Embeddings for the symbols
      generated from the decoder itself remain unchanged. This parameter has
      no effect if feed_previous=False.
    scope: VariableScope for the created subgraph; defaults to
      "embedding_rnn_decoder".

  Returns:
    A tuple of the form (outputs, state), where:
      outputs: A list of the same length as decoder_inputs of 2D Tensors with
        shape [batch_size x output_size] containing the generated outputs.
      state: The state of each decoder cell in each time-step. This is a list
        with length len(decoder_inputs) -- one item for each time-step.
        It is a 2D Tensor of shape [batch_size x cell.state_size].

  Raises:
    ValueError: When output_projection has the wrong shape.
  """
  with variable_scope.variable_scope(scope or "embedding_rnn_decoder") as scope:
    if output_projection is not None:
      #确定output_projection的形状是否符合要求
      dtype = scope.dtype
      proj_weights = ops.convert_to_tensor(output_projection[0], dtype=dtype)
      proj_weights.get_shape().assert_is_compatible_with([None, num_symbols])
      proj_biases = ops.convert_to_tensor(output_projection[1], dtype=dtype)
      proj_biases.get_shape().assert_is_compatible_with([num_symbols])

    #embedding矩阵
    embedding = variable_scope.get_variable("embedding",
                                            [num_symbols, embedding_size])

    loop_function = _extract_argmax_and_embed(
        embedding, output_projection,
        update_embedding_for_previous) if feed_previous else None
    #对于每一个解码器的输入都要进行wordembedding
    emb_inp = (
        embedding_ops.embedding_lookup(embedding, i) for i in decoder_inputs)
    #返回的还是rnn_decoder
    return rnn_decoder(emb_inp, initial_state, cell,
                       loop_function=loop_function)



def attention_decoder(decoder_inputs,
                      initial_state,
                      attention_states,
                      cell,
                      output_size=None,
                      num_heads=1,
                      loop_function=None,
                      dtype=None,
                      scope=None,
                      initial_state_attention=False):
  """基于注意力的RNN解码器
  具体来说，在解码的时候，RNN会查看注意力状态量，并且会集中在一些tensor上面
  In this context "attention" means that, during decoding, the RNN can look up
  information in the additional tensor attention_states, and it does this by
  focusing on a few entries from the tensor. This model has proven to yield
  especially good results in a number of sequence-to-sequence tasks. This
  implementation is based on http://arxiv.org/abs/1412.7449 (see below for
  details). It is recommended for complex sequence-to-sequence tasks.

  Args:
    decoder_inputs: 一个列表，其每个元素形状为[batch_size，input_size].
    initial_state: 一个列表，每个元素维度是[batch_size，cell.state_size].
    attention_states: 三维的tensor，并且后面两个维度必须明确指出，维度是[batch_size，attn_length，attn_size].
    cell: rnn_cell.RNNCell defining the cell function and size.
    output_size: Size of the output vectors; if None, we use cell.output_size.
    num_heads: 从attention状态中读取的线头数
    Number of attention heads that read from attention_states.
    loop_function: If not None, this function will be applied to i-th output
      in order to generate i+1-th input, and decoder_inputs will be ignored,
      except for the first element ("GO" symbol). This can be used for decoding,
      but also for training to emulate http://arxiv.org/abs/1506.03099.
      Signature -- loop_function(prev, i) = next
        * prev is a 2D Tensor of shape [batch_size x output_size],
        * i is an integer, the step number (when advanced control is needed),
        * next is a 2D Tensor of shape [batch_size x input_size].
    dtype: The dtype to use for the RNN initial state (default: tf.float32).
    scope: VariableScope for the created subgraph; default: "attention_decoder".
    initial_state_attention: If False (default), initial attentions are zero.
      If True, initialize the attentions from the initial state and attention
      states -- useful when we wish to resume decoding from a previously
      stored decoder state and attention states.

  Returns:
    A tuple of the form (outputs, state), where:
      outputs: A list of the same length as decoder_inputs of 2D Tensors of
        shape [batch_size x output_size]. These represent the generated outputs.
        Output i is computed from input i (which is either the i-th element
        of decoder_inputs or loop_function(output {i-1}, i)) as follows.
        First, we run the cell on a combination of the input and previous
        attention masks:
          cell_output, new_state = cell(linear(input, prev_attn), prev_state).
        Then, we calculate new attention masks:
          new_attn = softmax(V^T * tanh(W * attention_states + U * new_state))
        and then we calculate the output:
          output = linear(cell_output, new_attn).
      state: The state of each decoder cell the final time-step.
        It is a 2D Tensor of shape [batch_size x cell.state_size].

  Raises:
    ValueError: when num_heads is not positive, there are no inputs, shapes
      of attention_states are not set, or input size cannot be inferred
      from the input.
  """
  if not decoder_inputs:
    raise ValueError("Must provide at least 1 input to attention decoder.")
  if num_heads < 1:
    raise ValueError("With less than 1 heads, use a non-attention decoder.")
  if not attention_states.get_shape()[1:2].is_fully_defined():
    raise ValueError("Shape[1] and [2] of attention_states must be known: %s"
                     % attention_states.get_shape())
  if output_size is None:
    output_size = cell.output_size

  with tf.variable_scope(
      scope or "attention_decoder", dtype=dtype) as scope:
    dtype = scope.dtype
    batch_size = array_ops.shape(decoder_inputs[0])[0]  # Needed for reshaping.
    attn_length = attention_states.get_shape()[1].value
    attn_size = attention_states.get_shape()[2].value
    hidden = array_ops.reshape(attention_states, [-1, attn_length, 1, attn_size])
    hidden_features = []
    v = []
    attention_vec_size = attn_size  
    for a in xrange(num_heads):
      k = tf.get_variable("AttnW_%d" % a,[1, 1, attn_size, attention_vec_size])
      hidden_features.append(nn_ops.conv2d(hidden, k, [1, 1, 1, 1], "SAME"))
      v.append(tf.get_variable("AttnV_%d" % a, [attention_vec_size]))
    state = initial_state

    def attention(query):
      """Put attention masks on hidden using hidden_features and query."""
      ds = []  
      if nest.is_sequence(query):  
        query_list = nest.flatten(query)
        for q in query_list:  
          ndims = q.get_shape().ndims
          if ndims:
            assert ndims == 2
        query = array_ops.concat(1, query_list)
      for a in xrange(num_heads):
        with tf.variable_scope("Attention_%d" % a):
          y = linear(query, attention_vec_size, True)
          y = array_ops.reshape(y, [-1, 1, 1, attention_vec_size])
          s = math_ops.reduce_sum(
              v[a] * math_ops.tanh(hidden_features[a] + y), [2, 3])
          a = nn_ops.softmax(s)
          d = math_ops.reduce_sum(
              array_ops.reshape(a, [-1, attn_length, 1, 1]) * hidden,
              [1, 2])
          ds.append(array_ops.reshape(d, [-1, attn_size]))
      return ds

    outputs = []
    prev = None
    batch_attn_size = array_ops.pack([batch_size, attn_size])
    attns = [array_ops.zeros(batch_attn_size, dtype=dtype)
             for _ in xrange(num_heads)]
    for a in attns:  
      a.set_shape([None, attn_size])
    if initial_state_attention:
      attns = attention(initial_state)

    for i, inp in enumerate(decoder_inputs):
      if i > 0:
        tf.get_variable_scope().reuse_variables()
      if loop_function is not None and prev is not None:
        with tf.variable_scope("loop_function", reuse=True):
          inp = loop_function(prev, i)
      input_size = inp.get_shape().with_rank(2)[1]
      if input_size.value is None:
        raise ValueError("Could not infer input size from input: %s" % inp.name)
      x = linear([inp] + attns, input_size, True)
      cell_output, state = cell(x, state)
      if i == 0 and initial_state_attention:
        with tf.variable_scope(tf.get_variable_scope(),reuse=True):
          attns = attention(state)
      else:
        attns = attention(state)
      with tf.variable_scope("AttnOutputProjection"):
        output = linear([cell_output] + attns, output_size, True)
      if loop_function is not None:
        prev = output
      outputs.append(output)
  return outputs, state



def sequence_loss_by_example(logits, targets, weights,
                             average_across_timesteps=True,
                             softmax_loss_function=None, name=None):
  """Weighted cross-entropy loss for a sequence of logits (per example).

  Args:
    logits: List of 2D Tensors of shape [batch_size x num_decoder_symbols].
    targets: List of 1D batch-sized int32 Tensors of the same length as logits.
    weights: List of 1D batch-sized float-Tensors of the same length as logits.
    average_across_timesteps: If set, divide the returned cost by the total
      label weight.
    softmax_loss_function: Function (inputs-batch, labels-batch) -> loss-batch
      to be used instead of the standard softmax (the default if this is None).
    name: Optional name for this operation, default: "sequence_loss_by_example".

  Returns:
    1D batch-sized float Tensor: The log-perplexity for each sequence.

  Raises:
    ValueError: If len(logits) is different from len(targets) or len(weights).
  """
  if len(targets) != len(logits) or len(weights) != len(logits):
    raise ValueError("Lengths of logits, weights, and targets must be the same "
                     "%d, %d, %d." % (len(logits), len(weights), len(targets)))
  with ops.name_scope(name, "sequence_loss_by_example",
                      logits + targets + weights):
    log_perp_list = []
    for logit, target, weight in zip(logits, targets, weights):
      if softmax_loss_function is None:
        # TODO(irving,ebrevdo): This reshape is needed because
        # sequence_loss_by_example is called with scalars sometimes, which
        # violates our general scalar strictness policy.
        target = array_ops.reshape(target, [-1])
        crossent = nn_ops.sparse_softmax_cross_entropy_with_logits(
            logit, target)
      else:
        crossent = softmax_loss_function(logit, target)
      log_perp_list.append(crossent * weight)
    log_perps = math_ops.add_n(log_perp_list)
    if average_across_timesteps:
      total_size = math_ops.add_n(weights)
      total_size += 1e-12  # Just to avoid division by 0 for all-0 weights.
      log_perps /= total_size
  return log_perps


def sequence_loss(logits, targets, weights,
                  average_across_timesteps=True, average_across_batch=True,
                  softmax_loss_function=None, name=None):
  """Weighted cross-entropy loss for a sequence of logits, batch-collapsed.

  Args:
    logits: List of 2D Tensors of shape [batch_size x num_decoder_symbols].
    targets: List of 1D batch-sized int32 Tensors of the same length as logits.
    weights: List of 1D batch-sized float-Tensors of the same length as logits.
    average_across_timesteps: If set, divide the returned cost by the total
      label weight.
    average_across_batch: If set, divide the returned cost by the batch size.
    softmax_loss_function: Function (inputs-batch, labels-batch) -> loss-batch
      to be used instead of the standard softmax (the default if this is None).
    name: Optional name for this operation, defaults to "sequence_loss".

  Returns:
    A scalar float Tensor: The average log-perplexity per symbol (weighted).

  Raises:
    ValueError: If len(logits) is different from len(targets) or len(weights).
  """
  with ops.name_scope(name, "sequence_loss", logits + targets + weights):
    cost = math_ops.reduce_sum(sequence_loss_by_example(
        logits, targets, weights,
        average_across_timesteps=average_across_timesteps,
        softmax_loss_function=softmax_loss_function))
    if average_across_batch:
      batch_size = array_ops.shape(targets[0])[0]
      return cost / math_ops.cast(batch_size, cost.dtype)
    else:
      return cost


def model_with_buckets(encoder_inputs, decoder_inputs, targets, weights,
                       buckets, seq2seq, softmax_loss_function=None,
                       per_example_loss=False, name=None):
  """Create a sequence-to-sequence model with support for bucketing.

  The seq2seq argument is a function that defines a sequence-to-sequence model,
  e.g., seq2seq = lambda x, y: basic_rnn_seq2seq(x, y, rnn_cell.GRUCell(24))

  Args:
    encoder_inputs: A list of Tensors to feed the encoder; first seq2seq input.
    decoder_inputs: A list of Tensors to feed the decoder; second seq2seq input.
    targets: A list of 1D batch-sized int32 Tensors (desired output sequence).
    weights: List of 1D batch-sized float-Tensors to weight the targets.
    buckets: A list of pairs of (input size, output size) for each bucket.
    seq2seq: A sequence-to-sequence model function; it takes 2 input that
      agree with encoder_inputs and decoder_inputs, and returns a pair
      consisting of outputs and states (as, e.g., basic_rnn_seq2seq).
    softmax_loss_function: Function (inputs-batch, labels-batch) -> loss-batch
      to be used instead of the standard softmax (the default if this is None).
    per_example_loss: Boolean. If set, the returned loss will be a batch-sized
      tensor of losses for each sequence in the batch. If unset, it will be
      a scalar with the averaged loss from all examples.
    name: Optional name for this operation, defaults to "model_with_buckets".

  Returns:
    A tuple of the form (outputs, losses), where:
      outputs: The outputs for each bucket. Its j'th element consists of a list
        of 2D Tensors of shape [batch_size x num_decoder_symbols] (jth outputs).
      losses: List of scalar Tensors, representing losses for each bucket, or,
        if per_example_loss is set, a list of 1D batch-sized float Tensors.

  Raises:
    ValueError: If length of encoder_inputsut, targets, or weights is smaller
      than the largest (last) bucket.
  """
  if len(encoder_inputs) < buckets[-1][0]:
    raise ValueError("Length of encoder_inputs (%d) must be at least that of la"
                     "st bucket (%d)." % (len(encoder_inputs), buckets[-1][0]))
  if len(targets) < buckets[-1][1]:
    raise ValueError("Length of targets (%d) must be at least that of last"
                     "bucket (%d)." % (len(targets), buckets[-1][1]))
  if len(weights) < buckets[-1][1]:
    raise ValueError("Length of weights (%d) must be at least that of last"
                     "bucket (%d)." % (len(weights), buckets[-1][1]))

  all_inputs = encoder_inputs + decoder_inputs + targets + weights
  losses = []
  outputs = []
  with ops.name_scope(name, "model_with_buckets", all_inputs):
    for j, bucket in enumerate(buckets):
      with variable_scope.variable_scope(variable_scope.get_variable_scope(),
                                         reuse=True if j > 0 else None):
        bucket_outputs, _ = seq2seq(encoder_inputs[:bucket[0]],
                                    decoder_inputs[:bucket[1]])
        outputs.append(bucket_outputs)
        if per_example_loss:
          losses.append(sequence_loss_by_example(
              outputs[-1], targets[:bucket[1]], weights[:bucket[1]],
              softmax_loss_function=softmax_loss_function))
        else:
          losses.append(sequence_loss(
              outputs[-1], targets[:bucket[1]], weights[:bucket[1]],
              softmax_loss_function=softmax_loss_function))

  return outputs, losses
