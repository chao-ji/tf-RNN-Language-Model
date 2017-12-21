import os
import collections

import tensorflow as tf
import numpy as np

class DataFeeder(object):
  def __init__(self, config, filename, vocab=None):
    self.vocab = vocab
    self.raw_data = np.array(self.get_raw_data(filename))
    self.config = config
    self.epoch_size = ((len(self.raw_data) // self.config.batch_size) - 1) // self.config.num_steps

  def read_words(self, filename):
    with open(filename) as f:
      words = f.read().replace("\n", "<eos>").split()
    return words

  def file2ids(self, filename, word2id):
    words = self.read_words(filename)
    return [word2id[word] for word in words if word in word2id]

  def build_vocab(self, filename):
    words = self.read_words(filename)
    counter = collections.Counter(words)
    pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    words_sorted, _ = list(zip(*pairs))
    vocab = dict(zip(words_sorted, range(len(words_sorted))))
    return vocab

  def get_raw_data(self, filename):
    if self.vocab is None:
      self.vocab = self.build_vocab(filename)
    raw_data = self.file2ids(filename, self.vocab)
    return raw_data

  def batch_generator(self):
    raw_data = self.raw_data
    batch_size = self.config.batch_size
    num_steps = self.config.num_steps
    epoch_size = self.epoch_size
    batch_len = self.raw_data.shape[0] // self.config.batch_size
    data = raw_data[:batch_size*batch_len].reshape((batch_size, batch_len))

    if epoch_size <= 0:
      raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

    for i in xrange(epoch_size):
      x = data[:batch_size, i*num_steps:(i+1)*num_steps]
      y = data[:batch_size, i*num_steps+1:(i+1)*num_steps+1]
      yield x, y


class LanguageModel(object):
  def __init__(self, is_training, df, config):
    self.is_training = is_training
    self.df = df
    self.config = config

  def create_variables(self):
    inputs_batch = tf.placeholder(dtype=tf.int64, shape=[self.config.batch_size, self.config.num_steps])
    labels_batch = tf.placeholder(dtype=tf.int64, shape=[self.config.batch_size, self.config.num_steps])

    self.embedding = tf.get_variable("embedding", shape=[self.config.vocab_size,
                      self.config.embedding_size], dtype=tf.float32)
    self.softmax_w = tf.get_variable("softmax_w", shape=[self.config.embedding_size,
                      self.config.vocab_size], dtype=tf.float32)
    self.softmax_b = tf.get_variable("softmax_b", shape=[self.config.vocab_size],
                      dtype=tf.float32)
    return inputs_batch, labels_batch

  def inference(self, inputs_batch):
    inputs = tf.nn.embedding_lookup(self.embedding, inputs_batch)
    if self.is_training and self.config.keep_prob < 1:
      inputs = tf.nn.dropout(inputs, self.config.keep_prob)

    def make_cell(s):
      cell = tf.contrib.rnn.LSTMBlockCell(s, forget_bias=0.0)
      if self.is_training and self.config.keep_prob < 1:
        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.config.keep_prob)
      return cell

    sizes = [self.config.embedding_size] * self.config.num_layers
    multi_rnn_cell = tf.contrib.rnn.MultiRNNCell([make_cell(s) for s in sizes], state_is_tuple=True)
    self.initial_state = multi_rnn_cell.zero_state(self.config.batch_size, tf.float32)
    outputs, self.final_state = tf.nn.dynamic_rnn(multi_rnn_cell, inputs, initial_state=self.initial_state)
    outputs = tf.reshape(outputs, [-1, self.config.embedding_size])
    logits_batch = tf.nn.xw_plus_b(outputs, self.softmax_w, self.softmax_b)
    logits_batch = tf.reshape(logits_batch, [self.config.batch_size, self.config.num_steps, self.config.vocab_size])
    return logits_batch

  def loss(self, logits_batch, labels_batch):
    loss = tf.contrib.seq2seq.sequence_loss(
        logits_batch,
        labels_batch,
        tf.ones([self.config.batch_size, self.config.num_steps], dtype=tf.float32),
        average_across_timesteps=False,
        average_across_batch=True) 
    return tf.reduce_sum(loss)
    
  def get_learning_rate(self):
    lr = tf.Variable(0.0, trainable=False)
    new_lr = tf.placeholder(tf.float32, shape=[], name="new_learning_rate")
    update_op = tf.assign(lr, new_lr)
    return lr, new_lr, update_op

  def get_train_op(self, total_loss, lr):
    trainable_vars = tf.trainable_variables() 
    grads, _ = tf.clip_by_global_norm(tf.gradients(total_loss, trainable_vars),
                                      self.config.max_grad_norm)
    optimizer = tf.train.GradientDescentOptimizer(lr)
    train_op = optimizer.apply_gradients(zip(grads, trainable_vars),
                        global_step=tf.train.get_or_create_global_step())
    return train_op

  def update_lr(self, session, update_op, new_lr, lr_value):
    session.run(update_op, feed_dict={new_lr: lr_value})

  def run_epoch(self, sess, total_loss, df, inputs_batch, labels_batch,  eval_op=None):
    costs = 0.0
    iters = 0
    state = sess.run(self.initial_state)
    g = df.batch_generator()
    fetches = {"cost": total_loss, "final_state": self.final_state}

    if eval_op is not None:
      fetches["eval_op"] = eval_op

    for step, (inputs_batch_val, labels_batch_val) in enumerate(g):
      feed_dict = {}
      for i, (c, h) in enumerate(self.initial_state):
        feed_dict[c] = state[i].c
        feed_dict[h] = state[i].h

      feed_dict[inputs_batch] = inputs_batch_val
      feed_dict[labels_batch] = labels_batch_val

      vals = sess.run(fetches, feed_dict)
      cost = vals["cost"]
      state = vals["final_state"]

      costs += cost
      iters += self.config.num_steps

      if self.is_training and step % (df.epoch_size // 10) == 10:
        print("%.3f perplexity: %.3f" % (step * 1.0 / df.epoch_size, np.exp(costs / iters)))

    return np.exp(costs / iters)

  def build_graph(self):
    inputs_batch, labels_batch = self.create_variables()
    logits_batch = self.inference(inputs_batch)
    total_loss = self.loss(logits_batch, labels_batch)
    lr, new_lr, update_op = self.get_learning_rate()
    train_op = self.get_train_op(total_loss, lr)

    out_list = [total_loss, inputs_batch, labels_batch]
    if self.is_training:
      out_list.extend([train_op, lr, new_lr, update_op])
    return out_list

  def train(self, df, sess):
    total_loss, inputs_batch, labels_batch, train_op, lr, new_lr, update_op = self.build_graph()
    sess.run(tf.global_variables_initializer())

    for i in xrange(self.config.max_max_epoch):
      lr_decay = self.config.lr_decay ** max(i + 1 - self.config.max_epoch, 0.0)
      self.update_lr(sess, update_op, new_lr, self.config.init_lr * lr_decay)
      print("Epoch: %d Learning rate: %.3f" % (i + 1, sess.run(lr)))

      train_perplexity = self.run_epoch(sess, total_loss, df, inputs_batch, labels_batch, train_op)
      print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))

  def score(self, df, sess):
    total_loss, inputs_batch, labels_batch = self.build_graph()
    perplexity = self.run_epoch(sess, total_loss, df, inputs_batch, labels_batch)
    return perplexity
