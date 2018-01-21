import collections

import tensorflow as tf
import numpy as np

class DataFeeder(object):
  def __init__(self, batch_size, num_steps, filename, word_level=True, vocab=None, encoding="utf-8"):
    self.batch_size = batch_size
    self.num_steps = num_steps
    self.filename = filename 
    self.word_level = word_level
    self.vocab = vocab
    self.encoding = encoding

  def _read_items(self, filename):
    with open(filename) as f:
      try:
        items = f.read().decode(self.encoding)
      except UnicodeDecodeError as e:
        raise e
      if self.word_level:
        items = items.replace("\n", "<eos>").split()
    return items

  def _file_to_item_ids(self, filename, vocab):
    items = self._read_items(filename)
    return np.array([vocab[item] for item in items if item in vocab])

  def _build_vocab(self, filename):
    items = self._read_items(filename)
    counter = collections.Counter(items)
    item_count_pairs_sorted = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    self.index2item = list(zip(*item_count_pairs_sorted)[0])
    vocab = dict(zip(self.index2item, range(len(self.index2item))))
    return vocab

  def get_raw_data(self, filename):
    if self.vocab is None:
      self.vocab = self._build_vocab(filename)
    raw_data = self._file_to_item_ids(filename, self.vocab)
    return raw_data

  def batch_generator(self):
    if not hasattr(self, "raw_data"):
      self.raw_data = self.get_raw_data(self.filename)

      self.batch_len = len(self.raw_data) / self.batch_size
      self.num_epochs = (self.batch_len - 1) / self.num_steps
      self.data = self.raw_data[:self.batch_size*self.batch_len].reshape((self.batch_size, self.batch_len))

      if self.num_epochs <= 0:
        raise ValueError("`num_epochs` == 0, decrease `batch_size` or `num_steps`")

    for i in xrange(self.num_epochs):
      x = self.data[:self.batch_size, i*self.num_steps:(i+1)*self.num_steps]
      y = self.data[:self.batch_size, i*self.num_steps+1:(i+1)*self.num_steps+1]
      yield x, y


class MyBasicRNNCell(tf.contrib.rnn.RNNCell):
  """
  Subclassing `tf.contrib.rnn.RNNCell` and implementing `call`
  """
  def __init__(self, num_units, activation=None, reuse=None):
    super(MyBasicRNNCell, self).__init__(_reuse=reuse)
    
    self._num_units = num_units
    self._activation = activation or tf.tanh

  def call(self, inputs, state):
    inputs_and_state = tf.concat([inputs, state], axis=1)

    self._weights = tf.get_variable("rnn_kernel", [inputs_and_state.get_shape()[1],
      self._num_units], dtype=tf.float32)
    self._biases = tf.get_variable("rnn_bias", [self._num_units], dtype=tf.float32,
      initializer=tf.constant_initializer(value=0.0, dtype=tf.float32))
    affined = tf.matmul(inputs_and_state, self._weights) + self._biases
    output = new_state = self._activation(affined)
    return output, new_state

  @property
  def state_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units


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

    self._weights = tf.get_variable("lstm_kernel", [inputs_and_state.get_shape()[1], 
      4*self._num_units], dtype=tf.float32)
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


class RNNLanguageModel(object):
  """Language Model using vanilla rnn or lstm"""
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
                vocab_size=10000,
                cell_type="lstm",
                ckpt_path="/tmp/model.ckpt"):
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
    self.cell_type = cell_type
    self.ckpt_path = ckpt_path    

    self._initializer = tf.random_uniform_initializer(-self.init_scale, self.init_scale)

  def _get_placeholders(self, batch_shape, train_mode=True):
    inputs_batch = tf.placeholder(dtype=tf.int64, shape=batch_shape, name="inputs_batch")
    labels_batch = tf.placeholder(dtype=tf.int64, shape=batch_shape, name="labels_batch")
    if not train_mode:
      return inputs_batch, labels_batch
    new_lr = tf.placeholder(dtype=tf.float32, shape=[], name="new_learning_rate")
    return inputs_batch, labels_batch, new_lr

  def _get_variables(self, train_mode=True):
    embedding = tf.get_variable("embedding", shape=[self.vocab_size,
      self.embedding_size], dtype=tf.float32)
    softmax_w = tf.get_variable("softmax_w", shape=[self.embedding_size,
      self.vocab_size], dtype=tf.float32)
    softmax_b = tf.get_variable("softmax_b", shape=[self.vocab_size],
      dtype=tf.float32)
    if not train_mode:
      return embedding, softmax_w, softmax_b
    lr = tf.get_variable("learning_rate", shape=[], dtype=tf.float32,
      trainable=False)
    return embedding, softmax_w, softmax_b, lr

  def _make_cell(self, size, train_mode):
    if self.cell_type == "lstm":
      cell = MyBasicLSTMCell(size)
    elif self.cell_type == "rnn":
      cell = MyBasicRNNCell(size)
    cell = tf.contrib.rnn.DropoutWrapper(cell,
      output_keep_prob=self.keep_prob if train_mode else 1.0)
    return cell

  def _inference(self, embedding, softmax_w, softmax_b, inputs_batch, batch_shape, train_mode):
    batch_size, num_steps = batch_shape
    # [N, T, D]
    inputs = tf.nn.embedding_lookup(embedding, inputs_batch)
    inputs = tf.nn.dropout(inputs, self.keep_prob if train_mode else 1.0)
    multi_rnn_cell = tf.contrib.rnn.MultiRNNCell([self._make_cell(size, train_mode)
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

  def _build_graph(self, inputs_batch, labels_batch, batch_shape, train_mode=True):
    with tf.variable_scope("Model", initializer=self._initializer):
      if train_mode:
        embedding, softmax_w, softmax_b, lr = self._get_variables(train_mode)
      else:
        embedding, softmax_w, softmax_b = self._get_variables(train_mode)
      logits_batch, initial_state, final_state = self._inference(embedding, 
        softmax_w, softmax_b, inputs_batch, batch_shape, train_mode)

    loss = tf.contrib.seq2seq.sequence_loss(
        logits=logits_batch,
        targets=labels_batch,
        weights=tf.ones(batch_shape, dtype=tf.float32),
        average_across_timesteps=False,
        average_across_batch=True)

    if train_mode:
      return tf.reduce_sum(loss), initial_state, final_state, lr
    else:
      return tf.reduce_sum(loss), initial_state, final_state

  def _get_train_op(self, loss, lr):
    trainable_vars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss, trainable_vars),
                                      self.max_grad_norm)
    optimizer = tf.train.GradientDescentOptimizer(lr)
    train_op = optimizer.apply_gradients(zip(grads, trainable_vars),
                        global_step=tf.train.get_or_create_global_step())
    return train_op

  def _do_update_lr(self, sess, lr_val, lr, new_lr):
    sess.run(tf.assign(lr, new_lr), feed_dict={new_lr: lr_val})

  def _initialize_state(self, feed_dict, init_state, init_state_val):
    if self.cell_type == "lstm":
      for i, (c, h) in enumerate(init_state):
        feed_dict[c] = init_state_val[i].c
        feed_dict[h] = init_state_val[i].h
    elif self.cell_type == "rnn":
      for i, state in enumerate(init_state):
        feed_dict[state] = init_state_val[i]

  def _run_epoch(self, data_feeder, params, sess, train_mode=True, train_op=None):
    loss, inputs_batch, labels_batch, initial_state, final_state = params
    _test_mode = train_op is None

    total_loss, total_len, state = 0., 0, sess.run(initial_state)
    g = data_feeder.batch_generator()

    to_be_run = {"loss": loss, "final_state": final_state}
    if not _test_mode:
      to_be_run["train_op"] = train_op

    for step, (inputs_batch_val, labels_batch_val) in enumerate(g):
      feed_dict = {}
      self._initialize_state(feed_dict, initial_state, state)

      feed_dict[inputs_batch] = inputs_batch_val
      feed_dict[labels_batch] = labels_batch_val

      to_be_run_val = sess.run(to_be_run, feed_dict)
      loss, state = to_be_run_val["loss"], to_be_run_val["final_state"]

      total_loss += loss
      total_len += self.num_steps if train_mode else 1

      if not _test_mode and step % (data_feeder.num_epochs / 10) == 0:
        print("%.3f perplexity: %.3f" % (step * 1.0 / data_feeder.num_epochs, 
                                          np.exp(total_loss / total_len)))
    return np.exp(total_loss / total_len)

  def train(self, data_feeder):
    batch_shape = self.batch_size, self.num_steps

    graph = tf.Graph()
    with graph.as_default():   
      inputs_batch, labels_batch, new_lr = self._get_placeholders(batch_shape)
      loss, initial_state, final_state, lr = self._build_graph(inputs_batch, labels_batch, batch_shape)

      with tf.name_scope("Train_step"):
        train_op = self._get_train_op(loss, lr)
      gvi = tf.global_variables_initializer()

    with tf.Session(graph=graph) as sess:
      sess.run(gvi)  
 
      for i in xrange(self.max_max_epoch):
        lr_decay = self.lr_decay ** max(i + 1 - self.max_epoch, 0.0)
        self._do_update_lr(sess, self.init_lr * lr_decay, lr, new_lr)
        print("Epoch: %d Learning rate: %.3f" % (i + 1, sess.run(lr)))
        params = loss, inputs_batch, labels_batch, initial_state, final_state
        train_perplexity = self._run_epoch(data_feeder, params, sess, train_op=train_op)
        print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
    
      saver = tf.train.Saver()
      saver.save(sess, self.ckpt_path) 

  def score(self, data_feeder):
    batch_shape = 1, 1

    graph = tf.Graph()
    with graph.as_default():
      inputs_batch, labels_batch = self._get_placeholders(batch_shape, False)
      loss, initial_state, final_state = self._build_graph(inputs_batch, labels_batch, batch_shape, False)

    with tf.Session(graph=graph) as sess:
      saver = tf.train.Saver()
      saver.restore(sess, self.ckpt_path)

      params = loss, inputs_batch, labels_batch, initial_state, final_state
      perplexity = self._run_epoch(data_feeder, params, sess, train_mode=False)

    return perplexity

  def generate(self, primer, vocab, exclude_ids, length):
    batch_shape = 1, 1
    primer_ids = np.array([vocab[item] for item in primer if item in vocab])
    item_ids = np.arange(len(vocab))
    gen_ids = [primer_ids[0]]

    graph = tf.Graph()
    with graph.as_default():
      inputs_batch, _ = self._get_placeholders(batch_shape, False)
      with tf.variable_scope("Model"):
        embedding, softmax_w, softmax_b = self._get_variables(False)
        logits_batch, initial_state, final_state = self._inference(embedding,
          softmax_w, softmax_b, inputs_batch, batch_shape, False)
    
    with tf.Session(graph=graph) as sess:
      saver = tf.train.Saver()
      saver.restore(sess, self.ckpt_path)
      state_val = sess.run(initial_state)

      for i in xrange(1, len(primer_ids) + length):
        feed_dict = {inputs_batch: np.array(gen_ids[-1]).reshape(batch_shape)}
        self._initialize_state(feed_dict, initial_state, state_val)
        logits_batch_val, state_val = sess.run([logits_batch, final_state], feed_dict)
        next_id = primer_ids[i] if i < len(primer_ids) else \
          self._sample_next_id(item_ids, logits_batch_val[0, 0], exclude_ids)
        gen_ids.append(next_id)

    return gen_ids

  def _sample_next_id(self, item_ids, logits, exclude_ids):
    while True:
      id_ = np.random.choice(item_ids, p=self._softmax(logits))
      if id_ not in exclude_ids:
        return id_

  def _softmax(self, x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()
       
