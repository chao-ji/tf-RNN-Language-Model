import collections

import tensorflow as tf
import numpy as np

class DataFeeder(object):
  def __init__(self, batch_size, num_steps, filename, vocab=None):
    self.batch_size = batch_size
    self.num_steps = num_steps
    self.filename = filename 
    self.vocab = vocab

  def _read_words(self, filename):
    with open(filename) as f:
      words = f.read().replace("\n", "<eos>").split()
    return words

  def _file_to_word_ids(self, filename, vocab):
    words = self._read_words(filename)
    return [vocab[word] for word in words if word in vocab]

  def _build_vocab(self, filename):
    words = self._read_words(filename)
    counter = collections.Counter(words)
    word_count_pairs_sorted = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    words_sorted, _ = list(zip(*word_count_pairs_sorted))
    vocab = dict(zip(words_sorted, range(len(words_sorted))))
    return vocab

  def get_raw_data(self, filename):
    if self.vocab is None:
      self.vocab = self._build_vocab(filename)
    raw_data = self._file_to_word_ids(filename, self.vocab)
    return raw_data

  def batch_generator(self):
    if not hasattr(self, "raw_data"):
      self.raw_data = np.array(self.get_raw_data(self.filename))

      self.batch_len = len(self.raw_data) / self.batch_size
      self.epoch_size = (self.batch_len - 1) / self.num_steps
      self.data = self.raw_data[:self.batch_size*self.batch_len].reshape((self.batch_size, self.batch_len))

      if self.epoch_size <= 0:
        raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

    for i in xrange(self.epoch_size):
      x = self.data[:self.batch_size, i*self.num_steps:(i+1)*self.num_steps]
      y = self.data[:self.batch_size, i*self.num_steps+1:(i+1)*self.num_steps+1]
      yield x, y


class MyBasicLSTMCell(tf.contrib.rnn.RNNCell):
  """
  Subclassing `tf.contrib.rnn.RNNCell` and implementing `call`
  """
  def __init__(self, num_units, forget_bias=1.0, activation=None, reuse=None):
    super(MyBasicLSTMCell, self).__init__(_reuse=reuse)

    self._num_units = num_units
    self._forget_bias = forget_bias
    self._activation = activation or tf.tanh

  def call(self, inputs, state):
    c, h = state
    inputs_and_state = tf.concat([inputs, h], axis=1)

    self._weights = tf.get_variable("lstm_kernel", [inputs_and_state.get_shape()[1], 4*self._num_units],
        dtype=tf.float32)
    self._biases = tf.get_variable("lstm_bias", [4*self._num_units], dtype=tf.float32,
        initializer=tf.constant_initializer(value=0.0, dtype=tf.float32))

    affined = tf.matmul(inputs_and_state, self._weights) + self._biases
    i, j, f, o = tf.split(value=affined, num_or_size_splits=4, axis=1)

    new_c = c * tf.sigmoid(f + self._forget_bias) + tf.sigmoid(i) * self._activation(j)
    new_h = self._activation(new_c) * tf.sigmoid(o)

    new_state = tf.contrib.rnn.LSTMStateTuple(new_c, new_h)
    return new_h, new_state

  @property
  def state_size(self):
    return tf.contrib.rnn.LSTMStateTuple(self._num_units, self._num_units)

  @property
  def output_size(self):
    return self._num_units 


class LanguageModel(object):
  """Word-based Language Model"""
  def __init__(self,
                init_scale=0.1,
                init_lr=1.0,
                max_grad_norm=5.,
                num_layers=2,
                num_steps=20,
                embedding_size=200,
                hidden_sizes=(200, 200),
                max_epoch=4,
                max_max_epoch=13,
                keep_prob=1.0,
                lr_decay=0.5,
                batch_size=20,
                vocab_size=10000):
    self.init_scale = init_scale
    self.init_lr = init_lr
    self.max_grad_norm = max_grad_norm
    self.num_layers = num_layers
    self.num_steps = num_steps
    self.embedding_size = embedding_size
    self.hidden_sizes = hidden_sizes
    self.max_epoch = max_epoch
    self.max_max_epoch = max_max_epoch
    self.keep_prob = keep_prob
    self.lr_decay = lr_decay
    self.batch_size = batch_size
    self.vocab_size = vocab_size
    
    self._initializer = tf.random_uniform_initializer(-self.init_scale, self.init_scale)

  def _get_placeholders(self, batch_shape):
    inputs_batch = tf.placeholder(dtype=tf.int64, shape=batch_shape, name="inputs_batch")
    labels_batch = tf.placeholder(dtype=tf.int64, shape=batch_shape, name="labels_batch")
    if not hasattr(self, "_keep_prob"):
      self._keep_prob = tf.placeholder(dtype=tf.float32, shape=[], name="keep_prob")
    if not hasattr(self, "new_lr"):
      self.new_lr = tf.placeholder(dtype=tf.float32, shape=[], name="new_learning_rate")
    return inputs_batch, labels_batch

  def _get_variables(self):
    embedding = tf.get_variable("embedding", shape=[self.vocab_size,
      self.embedding_size], dtype=tf.float32)
    softmax_w = tf.get_variable("softmax_w", shape=[self.embedding_size,
      self.vocab_size], dtype=tf.float32)
    softmax_b = tf.get_variable("softmax_b", shape=[self.vocab_size],
      dtype=tf.float32)
    self.lr = tf.get_variable("learning_rate", shape=[], dtype=tf.float32,
      trainable=False)
    return embedding, softmax_w, softmax_b

  def inference(self, embedding, softmax_w, softmax_b, inputs_batch, batch_shape):
    batch_size, num_steps = batch_shape
    # [N, T, D]
    inputs = tf.nn.embedding_lookup(embedding, inputs_batch)
    inputs = tf.nn.dropout(inputs, self._keep_prob)
    def make_cell(size):
      cell = MyBasicLSTMCell(size)
      cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self._keep_prob)
      return cell
    multi_rnn_cell = tf.contrib.rnn.MultiRNNCell([make_cell(size) 
      for size in self.hidden_sizes], state_is_tuple=True)
    # [LSTMStateTuple(c=Tensor(shape=[N, D]), h=Tensor(shape=[N, D]))] * L
    initial_state = multi_rnn_cell.zero_state(batch_size, tf.float32)
    # [N, T, D]; [LSTMStateTuple(c=Tensor(shape=[N, D]), h=Tensor(shape=[N, D]))] * L
    outputs, final_state = tf.nn.dynamic_rnn(multi_rnn_cell, inputs, initial_state=initial_state)
    # [N*T, D] 
    outputs = tf.reshape(outputs, [-1, self.embedding_size])
    # [N*T, V]
    logits_batch = tf.matmul(outputs, softmax_w) + softmax_b
    # [N, T, V]
    logits_batch = tf.reshape(logits_batch, [batch_size, num_steps, self.vocab_size])
    self.cell = multi_rnn_cell
    return logits_batch, initial_state, final_state 

  def build_graph(self, inputs_batch, labels_batch, batch_shape):
    with tf.variable_scope("Model", reuse=tf.AUTO_REUSE, initializer=self._initializer):
      embedding, softmax_w, softmax_b = self._get_variables()
      logits_batch, initial_state, final_state = self.inference(embedding, 
        softmax_w, softmax_b, inputs_batch, batch_shape)

    loss = tf.contrib.seq2seq.sequence_loss(
        logits=logits_batch,
        targets=labels_batch,
        weights=tf.ones(batch_shape, dtype=tf.float32),
        average_across_timesteps=False,
        average_across_batch=True)
    return tf.reduce_sum(loss), initial_state, final_state

  def _get_train_step(self, loss):
    trainable_vars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss, trainable_vars),
                                      self.max_grad_norm)
    optimizer = tf.train.GradientDescentOptimizer(self.lr)
    train_step = optimizer.apply_gradients(zip(grads, trainable_vars),
                        global_step=tf.train.get_or_create_global_step())
    return train_step

  def _do_update_lr(self, sess, lr_val):
    sess.run(tf.assign(self.lr, self.new_lr), feed_dict={self.new_lr: lr_val})

  def run_epoch(self, data_feeder, graph, params, sess, train_step=None):
    loss, inputs_batch, labels_batch, initial_state, final_state = graph
    keep_prob, num_steps = params
    _test_mode = train_step is None

    total_loss, total_len, state = 0., 0, sess.run(initial_state)
    g = data_feeder.batch_generator()

    to_be_run = {"loss": loss, "final_state": final_state}
    if not _test_mode:
      to_be_run["train_step"] = train_step

    for step, (inputs_batch_val, labels_batch_val) in enumerate(g):
      feed_dict = {}
      for i, (c, h) in enumerate(initial_state):
        feed_dict[c] = state[i].c
        feed_dict[h] = state[i].h

      feed_dict[inputs_batch] = inputs_batch_val
      feed_dict[labels_batch] = labels_batch_val
      feed_dict[self._keep_prob] = keep_prob

      to_be_run_val = sess.run(to_be_run, feed_dict)
      loss, state = to_be_run_val["loss"], to_be_run_val["final_state"]

      total_loss += loss
      total_len += num_steps

      if not _test_mode and step % (data_feeder.epoch_size / 10) == 0:
        print("%.3f perplexity: %.3f" % (step * 1.0 / data_feeder.epoch_size, 
                                          np.exp(total_loss / total_len)))
    return np.exp(total_loss / total_len)

  def train(self, data_feeder, sess):
    batch_shape = self.batch_size, self.num_steps
    with tf.name_scope("Train_graph"):
      inputs_batch, labels_batch = self._get_placeholders(batch_shape)
      loss, initial_state, final_state = self.build_graph(inputs_batch, labels_batch, batch_shape)
    with tf.name_scope("train_step"):
      train_step = self._get_train_step(loss)
    sess.run(tf.global_variables_initializer())

    for i in xrange(self.max_max_epoch):
      lr_decay = self.lr_decay ** max(i + 1 - self.max_epoch, 0.0)
      self._do_update_lr(sess, self.init_lr * lr_decay)
      print("Epoch: %d Learning rate: %.3f" % (i + 1, sess.run(self.lr)))
      graph = loss, inputs_batch, labels_batch, initial_state, final_state
      params = self.keep_prob, self.num_steps
      train_perplexity = self.run_epoch(data_feeder, graph, params, sess, train_step)
      print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))

  def score(self, data_feeder, sess):
    batch_shape = 1, 1
    with tf.name_scope("Test_graph"):
      inputs_batch, labels_batch = self._get_placeholders(batch_shape)
      loss, initial_state, final_state = self.build_graph(inputs_batch, labels_batch, batch_shape)
    graph = loss, inputs_batch, labels_batch, initial_state, final_state
    params = 1.0, 1
    perplexity = self.run_epoch(data_feeder, graph, params, sess)
    return perplexity

  def generate(self, seed_words, vocab, index2word, exclude_ids, length, sess):
    batch_shape = 1, 1
    seed_ids = np.array([vocab[word] for word in seed_words if word in vocab])
    word_ids = np.arange(len(vocab))
    gen_ids = [seed_ids[0]]
    feed_dict = {self._keep_prob: 1.0}

    with tf.name_scope("Generate_graph"):
      inputs_batch, _ = self._get_placeholders(batch_shape)
      with tf.variable_scope("Model", reuse=tf.AUTO_REUSE):
        embedding, softmax_w, softmax_b = self._get_variables()
        logits_batch, initial_state, final_state = self.inference(embedding,
          softmax_w, softmax_b, inputs_batch, batch_shape)
    
    state_val = sess.run(initial_state)

    for i in xrange(1, len(seed_ids) + length):
      feed_dict[inputs_batch] = np.array(gen_ids[-1]).reshape(batch_shape)
      for l, (c, h) in enumerate(initial_state):
        feed_dict[c] = state_val[l].c
        feed_dict[h] = state_val[l].h
      logits_batch_val, state_val = sess.run([logits_batch, final_state], feed_dict)
      next_id = seed_ids[i] if i < len(seed_ids) else \
        self._sample_next_id(word_ids, logits_batch_val[0, 0], exclude_ids)
      gen_ids.append(next_id)

    return [index2word[id_] if id_ != 2 else "." for id_ in gen_ids]

  def _sample_next_id(self, word_ids, logits, exclude_ids):
    while True:
      id_ = np.random.choice(word_ids, p=self._softmax(logits))
      if id_ not in exclude_ids:
        return id_

  def _softmax(self, x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()
       
