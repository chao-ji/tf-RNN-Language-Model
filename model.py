import tensorflow as tf


class RNNLanguageModel(object):
  def __init__(self, hparams, dataset, mode):
    self._mode = mode

    initializer = tf.random_uniform_initializer(
        -hparams.init_scale, hparams.init_scale)
    tf.get_variable_scope().set_initializer(initializer)
    # [V, D]
    self._embed = _create_embed(hparams)
    self._output_layer = tf.layers.Dense(
        hparams.vocab_size, use_bias=True, name="output_projection")

    self.logits, self.loss, self.init_state, self.final_state = \
        self._build_graph(hparams, dataset, scope=None)

  @property
  def mode(self):
    return self._mode

  def _build_graph(self, hparams, dataset, scope):
    # [V, D]
    embed = self._embed
    # [N, T, D]
    inputs = tf.nn.embedding_lookup(embed, dataset.in_ids)
    if self.mode == tf.contrib.learn.ModeKeys.TRAIN and hparams.dropout > 0.0:
      inputs = tf.nn.dropout(inputs, hparams.dropout)

    cell = _create_rnn_cell(hparams, self.mode)

    init_state = cell.zero_state(dataset.batch_size, tf.float32)
    # outputs = [N, T, D]
    # final_state = [tuple(c=[N, D], h=[N, D])] * NUM_LAYERS
    outputs, final_state = tf.nn.dynamic_rnn(
        cell, inputs, initial_state=init_state)
    # [N, T, V]
    logits = self._output_layer(outputs)
    
    if self.mode != tf.contrib.learn.ModeKeys.INFER:
      loss = _compute_loss(logits, dataset)
    else:
      loss = tf.no_op()
    # logits = [N, T, V]
    # loss = scalar
    # init_state = [tuple(c=[N, D], h=[N, D])] * NUM_LAYERS
    # final_state = [tuple(c=[N, D], h=[N, D])] * NUM_LAYERS 
    return logits, loss, init_state, final_state

def _compute_loss(logits, dataset):
  loss = tf.reduce_mean(
      tf.nn.sparse_softmax_cross_entropy_with_logits(
          labels=dataset.out_ids, logits=logits), axis=0)
  # loss = scalar
  loss = tf.reduce_sum(loss)
  return loss

def _create_rnn_cell(hparams, mode):
  cell_list = [_single_cell(
      hparams.unit_type,
      hparams.num_units,
      hparams.forget_bias,
      hparams.dropout,
      mode) for _ in range(hparams.num_layers)]
  if len(cell_list) == 1:
    return cell_list[0]
  else:
    return tf.contrib.rnn.MultiRNNCell(cell_list)

def _single_cell(unit_type, num_units, forget_bias, dropout, mode):
  dropout = dropout if mode == tf.contrib.learn.ModeKeys.TRAIN else 0.0
  if unit_type == "lstm":
    single_cell = tf.contrib.rnn.BasicLSTMCell(
        num_units, forget_bias=forget_bias)
  elif unit_type == "rnn":
    single_cell = tf.contrib.rnn.BasicRNNCell(num_units)
  else:
    raise ValueError("Unknown unit_type: %s" % unit_type)

  if dropout > 0.0:
    single_cell = tf.contrib.rnn.DropoutWrapper(
        cell=single_cell, input_keep_prob=(1.0 - dropout))
  return single_cell

def _create_embed(hparams):
  return tf.get_variable("embed", shape=[hparams.vocab_size, hparams.num_units])
