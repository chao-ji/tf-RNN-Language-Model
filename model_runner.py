import tensorflow as tf
import numpy as np

import data
import model


class _BaseModelRunner(object):
  mode = None

  def __init__(self, builder, hparams):
    tf.contrib.learn.ModeKeys.validate(type(self).mode)
    self._graph = tf.Graph()
    with self._graph.as_default():

      self._dataset = data.LanguageModelDataset(hparams)
      self._tables_initializer = tf.tables_initializer()
      self._tables_initialized = False

      self._model = builder(hparams, self.dataset, type(self).mode)
      if type(self).mode == tf.contrib.learn.ModeKeys.TRAIN:
        self._iter_steps = tf.Variable(0, trainable=False, name="iter_steps")
      self._global_variables_initializer = tf.global_variables_initializer()

      self._params = tf.trainable_variables()
      self._saver = tf.train.Saver(self._params)

  @property
  def graph(self):
    return self._graph

  @property
  def dataset(self):
    return self._dataset

  @property
  def model(self):
    return self._model

  def restore_params_from(self, sess, ckpt_dir):
    _restore_params_from(self, sess, ckpt_dir)

  def persist_params_to(self, sess, ckpt):
    print("%s model is saving params to %s..." % (
        type(self).mode.upper(), ckpt))
    self._saver.save(sess, ckpt)


class RNNLanguageModelTrainer(_BaseModelRunner):
  mode = tf.contrib.learn.ModeKeys.TRAIN

  def __init__(self, builder, hparams):
    super(RNNLanguageModelTrainer, self).__init__(
        builder=builder,
        hparams=hparams)

    with self.graph.as_default():
      self.lr = self._get_lr(hparams)
      self.update_op = self._get_update_op(hparams)
    
  @property
  def iter_steps(self):
    return self._iter_steps

  def _get_lr(self, hparams):
    lr = tf.constant(hparams.lr)
    lr_decay = tf.pow(hparams.lr_decay, tf.cast(tf.maximum(
        self.iter_steps + 1 - hparams.max_epoch,
        0), tf.float32))
    return lr_decay * lr

  def _get_update_op(self, hparams):
    opt = tf.train.GradientDescentOptimizer(self.lr)
    params = self._params
    gradients = tf.gradients(self.model.loss, params)

    clipped_grads, _ = tf.clip_by_global_norm(gradients, hparams.max_grad_norm)
    update_op = opt.apply_gradients(zip(clipped_grads, params))

    return update_op

  def train(self, sess, hparams, state):
    feed_dict = _initialize_state(self, hparams, state)     

    _, loss, state, lr = sess.run([
        self.update_op, 
        self.model.loss,
        self.model.final_state,
        self.lr], feed_dict)

    return loss, state, lr

  def update_iter_steps(self, sess, iter_steps):
    sess.run(tf.assign(self.iter_steps, iter_steps)) 


class RNNLanguageModelEvaluator(_BaseModelRunner):
  mode = tf.contrib.learn.ModeKeys.EVAL

  def eval(self, sess, hparams, state):
    feed_dict = _initialize_state(self, hparams, state)

    loss, state = sess.run([
        self.model.loss,
        self.model.final_state], feed_dict)

    return loss, state


class SequenceGenerator(object):
  mode = tf.contrib.learn.ModeKeys.INFER

  def __init__(self, builder, hparams):
    tf.contrib.learn.ModeKeys.validate(type(self).mode)
    self._graph = tf.Graph()
    with self._graph.as_default():

      self._dataset = data.GeneratorInput(hparams)
      self._tables_initializer = tf.tables_initializer()
      self._tables_initialized = False

      self._model = builder(hparams, self.dataset, type(self).mode)

      self._params = tf.trainable_variables()
      self._saver = tf.train.Saver(self._params)

  @property
  def graph(self):
    return self._graph

  @property
  def dataset(self):
    return self._dataset

  @property
  def model(self):
    return self._model

  def restore_params_from(self, sess, ckpt_dir):
    _restore_params_from(self, sess, ckpt_dir)

  def generate_text(self, sess, hparams, primer_seq, length, exclude_ids):
    item_ids = np.arange(hparams.vocab_size)

    state = sess.run(self.model.init_state)
    inputs = primer_seq

    out_list = []

    for i in range(length):
      feed_dict = _initialize_state(self, hparams, state)
      feed_dict[self.dataset.input_text] = inputs
      state, logits = sess.run([self.model.final_state, self.model.logits], feed_dict)

      next_id = _sample_next_id(item_ids, logits[0, -1], exclude_ids)
      out_list.append(self.dataset.reverse_vocab[next_id])
      inputs = self.dataset.reverse_vocab[next_id]
    return out_list

def _sample_next_id(item_ids, logits, exclude_ids):
  while True:
    id_ = np.random.choice(item_ids, p=_softmax(logits))
    if id_ not in exclude_ids:
      return id_

def _softmax(x):
  ex = np.exp(x - np.max(x))
  return ex / ex.sum()

def _restore_params_from(runner, sess, ckpt_dir):
  if not runner._tables_initialized:
    sess.run(runner._tables_initializer)
    runner._tables_initialized = True

  latest_ckpt = tf.train.latest_checkpoint(ckpt_dir)
  if latest_ckpt:
    print("%s model is loading params from %s..." % (
        type(runner).mode.upper(), latest_ckpt))
    runner._saver.restore(sess, latest_ckpt)
  else:
    print("%s model is creating fresh params..." %
        type(runner).mode.upper())
    sess.run(runner._global_variables_initializer)

def _initialize_state(runner, hparams, state):
  feed_dict = {}
  if hparams.unit_type == "lstm":
    for i, (c, h) in enumerate(runner.model.init_state):
      feed_dict[c] = state[i].c
      feed_dict[h] = state[i].h
  elif hparams.unit_type == "rnn":
    for i, state in enumerate(self.model.init_state):
      feed_dict[state] = state[i]
  return feed_dict

def compute_init_state(runner, sess):
  return sess.run(runner.model.init_state)
