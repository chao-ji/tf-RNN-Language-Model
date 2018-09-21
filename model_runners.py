import tensorflow as tf
import numpy as np


class RNNLanguageModelTrainer(object):
  """RNN language model trainer.

  `train()` method adds training related ops to the graph and outputs a 
  tensor_dict holding ops/tensors to be run in a `tf.Session`.
  """
  def __init__(self, 
               prediction_model,
               batch_size,
               init_lr,
               lr_decay,
               max_grad_norm,
               num_epochs_no_decay):
    """Constructor.

    Args:
      prediction_model: a RNNLanguagePredictionModel instance.
      batch_size: int scalar, batch size
      init_lr: float scalar, initial learning rate.
      lr_decay: float scalar, learning rate is decayed by a factor of 
        `lr_decay`.
      max_grad_norm: float scalar, maximum gradient norm.
      num_epochs_no_decay: int scalar, the learning rate remains for the first
        `num_epochs_no_decay` epochs.
    """
    self._prediction_model = prediction_model
    self._batch_size = batch_size
    self._init_lr = init_lr
    self._lr_decay = lr_decay
    self._max_grad_norm = max_grad_norm
    self._num_epochs_no_decay = num_epochs_no_decay

    self._epoch_index = None

  def train(self, filenames, dataset):
    """Adds training related ops to the graph.

    Args:
      filenames: a list of strings, file paths of training data.
      dataset: an RNNLanguageModelDataset instance.

    Returns:
      to_be_run_dict: dict mapping from names to tensors/operations, holding
        the following entries:
        { 'grad_update_op': optimization ops,
          'loss': cross entropy loss,
          'learning_rate': float-scalar learning rate,
          'final_state': list of state tuple, the final hidden state or RNN.}
    """
    tensor_dict = dataset.get_tensor_dict(filenames)
    logits, final_state = self._prediction_model.predict(
        tensor_dict['inputs'], self._batch_size) 
    loss = _create_loss(logits, tensor_dict['labels'])

    optimizer, learning_rate = self._build_optimizer()     
    grads_and_vars = optimizer.compute_gradients(
        loss, colocate_gradients_with_ops=True)
    grads_and_vars = list(zip(*grads_and_vars))
    grads, vars_ = grads_and_vars[0], grads_and_vars[1]
    clipped_grads, _ = tf.clip_by_global_norm(grads, self._max_grad_norm)
    grad_update_op = optimizer.apply_gradients(zip(clipped_grads, vars_))
    to_be_run_dict = {'grad_update_op': grad_update_op, 'loss': loss,
        'learning_rate': learning_rate, 'final_state': final_state}
    return to_be_run_dict
  
  def _build_optimizer(self):
    """Builds learning rate tensor and optimizer

    The learning rate starts at `init_lr` and lasts for `num_epochs_no_decay` 
    epochs and will be decayed by a factor of `lr_decay` for each epoch onwards.
    
    Returns:
      optimizer: an Optimizer instance.
      lr: float scalar tensor, learning rate.
    """
    self._epoch_index = tf.train.get_or_create_global_step()
    lr = tf.constant(self._init_lr)
    lr_decay = tf.pow(self._lr_decay, tf.to_float(tf.maximum(
        self._epoch_index + 1 - self._num_epochs_no_decay, 0)))   
    lr *= lr_decay
    optimizer = tf.train.GradientDescentOptimizer(lr)
    return optimizer, lr

  def increment_epoch_index(self, sess):
    """Increments the non-trainable variable `epoch_index`."""
    sess.run(tf.assign(self._epoch_index, self._epoch_index + 1))


class RNNLanguageModelEvaluator(object):
  """RNN language model trainer.

  `evaluate()` method adds evaluation related ops to the graph and outputs a 
  tensor_dict holding ops/tensors to be run in a `tf.Session`."""
  def __init__(self, prediction_model, batch_size):
    """Constructor.

    Args:
      prediction_model: an RNNLanguagePredictionModel instance.
      batch_size: int scalar, batch size
    """
    self._prediction_model = prediction_model
    self._batch_size = batch_size

  def evaluate(self, filenames, dataset):
    """Adds evaluation related ops to the graph.

    Args:
      filenames: a list of strings, file paths of training data.
      dataset: an RNNLanguageModelDataset instance.

    Returns:
      to_be_run_dict: dict mapping from names to tensors/operations, holding
        the following entries:
        { 'loss': cross entropy loss,
          'final_state': a list of state tuple or a single state tuple}
    """
    tensor_dict = dataset.get_tensor_dict(filenames)
    logits, final_state = self._prediction_model.predict(
        tensor_dict['inputs'], self._batch_size)
    loss = _create_loss(logits, tensor_dict['labels'])
    to_be_run_dict = {'loss': loss, 'final_state': final_state} 
    return to_be_run_dict


class RNNLanguageModelSeqGenerator(object):
  """RNN language model sequence generator.

  `generate()` method adds ops related to sequence generation to the graph and
  outputs a tensor_dict holding ops/tensors to be run in a `tf.Session`.
  """
  def __init__(self, prediction_model, gen_seq_len, use_sampling=True):
    """Constructor.

    Args:
      prediction_model: an RNNLanguagePredictionModel instance.
      gen_seq_len: int scalar, the length of sequence to be generated.
      use_sampling: bool scalar, whether to use sampling strategy (default)
        or argmax strategy for the `self._generate_basic()`.
    """
    self._prediction_model = prediction_model
    self._gen_seq_len = gen_seq_len
    self._use_sampling = use_sampling

    self._unk_offset = 1

  def generate(self, filename, dataset):
    """Adds ops related to sequence generation to the graph.

    Args:
      filename: string scalar, path to the text file containing primer seqs.
      dataset: an RNNLanguageModelDataset instance.

    Returns:
      to_be_run_dict: dict mapping from names to tensors/operations, holding
        the following entries:
        { 'gen_seq': int tensor of shape [gen_seq_len] }
    """
    return self._generate_basic(filename, dataset)

  def _generate_basic(self, filename, dataset):
    """Basic sequence generator. Use either argmax strategy (the word that has
    the largest logit) or sampling strategy. 
    """
    cell = self._prediction_model.cell
    proj = self._prediction_model.projection_layer
    rev_vocab_table = dataset.get_rev_vocab_table()

    tensor_dict = dataset.get_tensor_dict(filename)
    primer = tensor_dict['primer']
    primer_len = tf.size(primer)    

    with tf.variable_scope('Predict'):
      embed = self._prediction_model.create_embedding() 

      def step_fn(step, index, state, ta):
        def argmax(logits):
          """Deterministically takes the word with largest logit as the
          next token.
          """
          return tf.to_int64(
              tf.argmax(logits[self._unk_offset:]) + self._unk_offset)

        def sample(logits):
          """Samples the next token according to `logits`. If the sampled word
          happens to be `<unk>` (index == 0), falls back to `argmax`.
          """
          next_index = tf.distributions.Categorical(logits=logits).sample() 
          return tf.cond(tf.equal(next_index, 0), 
              lambda: argmax(logits), lambda: tf.to_int64(next_index))

        inputs = tf.gather(embed, tf.expand_dims(index, axis=0))
        with tf.variable_scope('rnn'):
          outputs, next_state = cell(inputs, state)
        logits = proj(outputs)[0]
        next_index = tf.cond(step < primer_len - 1, lambda: primer[step + 1], 
            lambda: sample(logits) if self._use_sampling else argmax(logits))
        return step + 1, next_index, next_state, ta.write(step, index)

      size = primer_len + self._gen_seq_len
      init_state = cell.zero_state(1, tf.float32)
      init_ta = tf.TensorArray(tf.int64, size=size)

      _, _, _, result_array = tf.while_loop(
          lambda step, index, state, ta: step < size, 
          step_fn, [0, primer[0], init_state, init_ta], back_prop=False)
      gen_seq = rev_vocab_table.lookup(result_array.stack())
    return {'gen_seq': gen_seq}


def _create_loss(logits, labels, scope=None):
  """Creates ops that lead from `logits` and `labels` to the loss tensor.

  Args:
    logits: float tensor of shape [batch_size, num_steps, vocab_size].
    labels: float tensor of shape [batch_size, num_steps].
    scope: string scalar, scope name.

  Returns:
    loss: float scalar tensor, loss.
  """
  with tf.name_scope(scope, 'Loss', [logits, labels]):
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits), axis=0)
    loss = tf.reduce_sum(loss)
    return loss


def eval_init_state(model_runner, sess):
  """Evaluate initial state of RNN Cell in a `tf.Session`.

  Args:
    model_runner: an RNNLanguageModelTrainer or RNNLanguageModelEvaluator 
      instance.
    sess: a `tf.Session` instance.    

  Returns:
    a list of state tuple holding initial hidden state values.
  """
  return sess.run(model_runner._prediction_model.init_state)


def get_init_state_to_value_feed_dict(model_runner, state_val):
  """Generates the feed dict mapping from tensor names to values to initialize
  the hidden states of RNN.

  Args:
    model_runner: an RNNLanguageModelTrainer or RNNLanguageModelEvaluator 
      instance.
    state_val: a list of state tuples that have the same shape as the initial
      state of RNN Cell.

  Returns:
    feed_dict: a dict mapping from RNN state tensor names to values.
  """
  feed_dict = {}
  if model_runner._prediction_model._unit_type == 'lstm':
    for i, (c, h) in enumerate(model_runner._prediction_model.init_state):
      feed_dict[c] = state_val[i].c
      feed_dict[h] = state_val[i].h
  elif model_runner._prediction_model._unit_type == 'rnn':
    for i, state in enumerate(model_runner._prediction_model.init_state):
      feed_dict[state] = state_val[i]  
  return feed_dict

