import tensorflow as tf

import data
import model


class _BaseModelRunner(object):
  mode = None

  def __init__(self, builder, hparams):
    tf.contrib.learn.ModeKeys.validate(type(self).mode)
    self._graph = tf.Graph()
    with self._graph.as_default():
      self._dataset = data.SequenceDataset(hparams)

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
    if not self._tables_initialized:
      sess.run(self._tables_initializer)
      self._tables_initialized = True

    latest_ckpt = tf.train.latest_checkpoint(ckpt_dir)
    if latest_ckpt:
      print("%s model is loading params from %s..." % (
          type(self).mode.upper(), latest_ckpt))
      self._saver.restore(sess, latest_ckpt)
    else:
      print("%s model is creating fresh params..." % 
          type(self).mode.upper())
      sess.run(self._global_variables_initializer)

  def persist_params_to(self, sess, ckpt):
    print("%s model is saving params to %s..." % (
        type(self).model.upper(), ckpt))
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

  def _initialize_state(self, hparams, state_val):
    feed_dict = {}
    if hparams.unit_type == "lstm":
      for i, (c, h) in enumerate(self.model.init_state):
        feed_dict[c] = state_val[i].c
        feed_dict[h] = state_val[i].h
    elif hparams.unit_type == "rnn":
      for i, state in enumerate(self.model.init_state):
        feed_dict[state] = state_val[i]

    return feed_dict

  def train(self, sess, hparams, state_val):
    feed_dict = self._initialize_state(hparams, state_val)     

    _, loss, state_val = sess.run([
        self.update_op, 
        self.model.loss,
        self.model.final_state], feed_dict)

    return loss, state_val 



