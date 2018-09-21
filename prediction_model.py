import tensorflow as tf


class RNNLanguagePredictionModel(object):
  """RNN language prediction model.

  Call `predict()` method to generate prediction logit tensor and final hidden
  states from input tensor."""
  def __init__(self, 
               vocab_size,
               num_units,
               unit_type='lstm',
               forget_bias=1.0,
               keep_prob=1.0,
               num_layers=2,
               init_scale=0.1,
               is_training=False):
    """Constructor.

    Args:
      vocab_size: int scalar, num of items in vocabulary.
      num_units: int scalar, the num of units in an RNN Cell and the word
        embedding size.
      unit_type: string scalar, the type of RNN cell ('lstm', 'rnn').
      forget_bias: float scalar, forget bias in LSTM Cell. Defaults to 1.0.
      keep_prob: float scalar, dropout rate equals `1 - keep_prob`.
      num_layers: int scalar, num of RNN layers.
      init_scale: float scalar, the random uniform initializer has range
        [-init_scale, init_scale].
      is_training: bool scalar, whether prediction model is in training mode.
    """
    self._vocab_size = vocab_size
    self._num_units = num_units
    self._unit_type = unit_type
    self._forget_bias = forget_bias
    self._keep_prob = keep_prob
    self._num_layers = num_layers
    self._init_scale = init_scale
    self._is_training = is_training

    self._init_state = None
    self._cell = self._build_rnn_cell() 
    self._projection_layer = tf.layers.Dense(self._vocab_size, use_bias=True, 
        name='projection_layer')

  @property
  def init_state(self):
    return self._init_state

  @property
  def cell(self):
    return self._cell

  @property
  def projection_layer(self):
    return self._projection_layer

  def predict(self, inputs, batch_size, scope=None):
    """Generates the prediction logit tensor from input tensor. Has the side
    effect of setting attributes `init_state` and `final_state`.

    Args:
      inputs: int tensor of shape [batch_size, num_steps], holding the input
        word indices.
      batch_size: scalar int tensor or scalar int, batch size.
      scope: string scalar, scope name.
    
    Returns:
      logits: float tensor of shape [batch_size, num_steps, vocab_size]
      final_state: a list of state tuples (or just a tensor) of shape 
        [batch_size, num_units]
    """
    initializer = tf.random_uniform_initializer(-self._init_scale, 
        self._init_scale)
    with tf.variable_scope(scope, 'Predict', [inputs], initializer=initializer):
      self._embedding = embedding = self.create_embedding()
      inputs = tf.nn.embedding_lookup(embedding, inputs)

      if self._is_training and self._keep_prob < 1.0:
        inputs = tf.nn.dropout(inputs, keep_prob=self._keep_prob)
    
      init_state = self._cell.zero_state(batch_size, tf.float32)
      outputs, final_state = tf.nn.dynamic_rnn(
          self._cell, inputs, initial_state=init_state)
      logits = self._projection_layer(outputs)  
      self._init_state = init_state
      return logits, final_state
 
  def create_embedding(self):
    """Creates embedding matrix [vocab_size, num_units] for a vocabulary.

    Returns:
      embedding: float tensor with shape [vocab_size, num_units]
    """
    return tf.get_variable('embedding', shape=[self._vocab_size, 
        self._num_units])

  def _build_rnn_cell(self):
    """Returns an RNN Cell instance."""
    cell_list = [self._create_single_cell() for _ in range(self._num_layers)]
    if len(cell_list) == 1:
      return cell_list[0]
    else:
      return tf.contrib.rnn.MultiRNNCell(cell_list)

  def _create_single_cell(self):
    """Returns an RNN Cell instance."""
    if self._unit_type == 'lstm':
      single_cell = tf.contrib.rnn.BasicLSTMCell(
          self._num_units, forget_bias=self._forget_bias)
    elif self._unit_type == 'rnn':
      single_cell = tf.contrib.rnn.BasicRNNCell(self._num_units)
    else:
      raise ValueError('Unknown unit type: %s' % self._unit_type)

    if self._is_training and self._keep_prob < 1.0:
      single_cell = tf.contrib.rnn.DropoutWrapper(
          cell=single_cell, output_keep_prob=self._keep_prob)
    return single_cell

