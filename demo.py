import os

import data
import model
import model_runner

import numpy as np
import tensorflow as tf


word_level = False
num_steps = 50
num_units = 128 
vocab_size = 50
lr = 0.1

hparams = tf.contrib.training.HParams(
  word_level=word_level,
  file_data = "/home/chaoji/Desktop/lstm_language_model/simple-examples/data/ptb.train.txt",
  file_vocab = "/home/chaoji/Desktop/lstm_language_model/simple-examples/data/ptb.train.txt",

  unit_type = "lstm",
  forget_bias = 1.0,
  init_scale = 0.1,
  lr = lr,
  max_grad_norm = 5.,
  num_layers = 2,
  num_steps = num_steps,
  num_units = num_units,
  max_epoch = 4,
  max_max_epoch = 13,
  dropout = 0.0,
  lr_decay = 0.5,
  batch_size = 20,
  vocab_size = vocab_size)

hparams_eval = tf.contrib.training.HParams(
  word_level=word_level,
  file_data = "/home/chaoji/Desktop/lstm_language_model/simple-examples/data/ptb.test.txt",
  file_vocab = "/home/chaoji/Desktop/lstm_language_model/simple-examples/data/ptb.train.txt",

  unit_type = "lstm",
  forget_bias = 1.0,
  init_scale = 0.1,
  num_layers = 2,
  num_steps = num_steps,
  num_units = num_units,
  dropout = 0.0,
  batch_size = 20,
  vocab_size = vocab_size)

hparams_infer = tf.contrib.training.HParams(
  word_level=word_level,
  file_vocab = "/home/chaoji/Desktop/lstm_language_model/simple-examples/data/ptb.train.txt", 
  unit_type = "lstm",
  forget_bias = 1.0,
  init_scale = 0.1,
  num_layers = 2,
  num_units = num_units,
  dropout = 0.0,
  vocab_size = vocab_size)


def train_epoch(hparams, trainer, train_sess, i):
  trainer.dataset.init_iterator(train_sess)

  state_val = model_runner.compute_init_state(trainer, train_sess)
  total_loss = 0.0
  total_word_count = 0.0

  trainer.update_iter_steps(train_sess, i)
  while True:
    try:
      loss, state_val, lr = trainer.train(train_sess, hparams, state_val)

      total_loss += loss
      total_word_count += hparams.num_steps
    except tf.errors.OutOfRangeError:
      train_ppl = np.exp(total_loss / total_word_count)
      print("epoch: %d, train ppl: %.2f, lr: %f" % (i, train_ppl, lr))
      break
  return train_ppl

def eval_epoch(hparams, evaluator, eval_sess, i):
  evaluator.dataset.init_iterator(eval_sess)

  state_val = model_runner.compute_init_state(evaluator, eval_sess)
  total_loss = 0.0
  total_word_count = 0.0

  while True:
    try:
      loss, state_val = evaluator.eval(eval_sess, hparams, state_val)

      total_loss += loss
      total_word_count += hparams.num_steps
    except tf.errors.OutOfRangeError:
      eval_ppl = np.exp(total_loss / total_word_count)
      print("epoch: %d, eval ppl: %.2f" % (i, eval_ppl))
      break
  return eval_ppl


ckpt = "/home/chaoji/Dropbox/tensorflow/rnn_language_model/"

builder = model.RNNLanguageModel
trainer = model_runner.RNNLanguageModelTrainer(builder, hparams)
evaluator = model_runner.RNNLanguageModelEvaluator(builder, hparams_eval)

train_sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True), graph=trainer.graph)
eval_sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True), graph=evaluator.graph)

trainer.restore_params_from(train_sess, ckpt)

for i in range(13):
  train_ppl = train_epoch(hparams, trainer, train_sess, i)
  trainer.persist_params_to(train_sess, os.path.join(ckpt, "lm"))

  evaluator.restore_params_from(eval_sess, ckpt)
  eval_ppl = eval_epoch(hparams_eval, evaluator, eval_sess, i)
  print

generator = model_runner.SequenceGenerator(builder, hparams_infer)
with tf.Session(graph=generator.graph) as sess:
  generator.restore_params_from(sess, ckpt)
  a = generator.generate_text(sess, hparams_infer, "It seems that", 20, set([1]))
