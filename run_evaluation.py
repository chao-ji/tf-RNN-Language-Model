r"""Executable for evaluating a trained language model.

Example:
  python run_evaluation.py \
      --filenames=/PATH/TO/FILENAMES \
      --vocab_filenames=/PATH/TO/VOCAB_FILENAMES \ 
      --ckpt_path=/PATH/TO/CKPT
"""
import tensorflow as tf
import numpy as np

from dataset import LanguageModelDataset
from prediction_model import RNNLanguagePredictionModel
from model_runners import RNNLanguageModelEvaluator
from model_runners import eval_init_state
from model_runners import get_init_state_to_value_feed_dict

flags = tf.app.flags


flags.DEFINE_boolean('char_level', False, 'Language model is at char level '
    '(Default) or word level.')
flags.DEFINE_integer('batch_size', 20, 'Batch size of each mini batch.')
flags.DEFINE_integer('num_steps', 35, 'Num of time steps of each mini batch.')
flags.DEFINE_integer('max_vocab_size', 0, 'Maximum vocabulary size. If > 0, '
    'the top `max_vocab_size` most frequent words are kept in vocabulary.')
flags.DEFINE_integer('min_count', 0, 'Words whose counts < `min_count` are not '
    'included in the vocabulary.')
flags.DEFINE_boolean('append_period', True, 'Whether to append "." to each '
    'line of text.')

flags.DEFINE_integer('num_units', 650, 'The num of units in an RNN Cell and '
    'the word embedding size.')
flags.DEFINE_string('unit_type', 'lstm', 'The type of RNN cell'
    ' ("lstm", "rnn").')
flags.DEFINE_float('forget_bias', 0.0, 'Forget bias in LSTM Cell. ')
flags.DEFINE_integer('num_layers', 2, 'Num of RNN layers.')

flags.DEFINE_string('ckpt_path', None, 'Path to the checkpoint file.')
flags.DEFINE_list('vocab_filenames', None, 'Comma-separated list of names of '
    'text files to build vocabulary from.')
flags.DEFINE_list('filenames', None, 'Comma-separated list of names of input '
    'text files. Could be different from `vocab_filenames`.')

FLAGS = flags.FLAGS


def main(_):
  dataset = LanguageModelDataset(char_level=FLAGS.char_level,
                                 batch_size=FLAGS.batch_size,
                                 num_steps=FLAGS.num_steps,
                                 max_vocab_size=FLAGS.max_vocab_size,
                                 min_count=FLAGS.min_count,
                                 append_period=FLAGS.append_period)
  dataset.build_vocab(FLAGS.vocab_filenames)

  prediction_model = RNNLanguagePredictionModel(
      vocab_size=len(dataset._table_words), 
      num_units=FLAGS.num_units,
      unit_type=FLAGS.unit_type,
      forget_bias=FLAGS.forget_bias,
      num_layers=FLAGS.num_layers,
      is_training=False)

  evaluator = RNNLanguageModelEvaluator(
      prediction_model=prediction_model, 
      batch_size=FLAGS.batch_size)

  to_be_run_dict = evaluator.evaluate(FLAGS.filenames, dataset)
  saver = tf.train.Saver()

  sess = tf.InteractiveSession()
  sess.run(tf.tables_initializer())
  sess.run(tf.global_variables_initializer())
  sess.run(dataset.iterator_initializer)
  saver.restore(sess, FLAGS.ckpt_path)
  state_val = eval_init_state(evaluator, sess)

  total_loss = 0 
  word_count = 0

  while True:
    try:
      feed_dict = get_init_state_to_value_feed_dict(evaluator, state_val)
      result_dict = sess.run(to_be_run_dict, feed_dict=feed_dict)
      state_val = result_dict['final_state']

      total_loss += result_dict['loss']
      word_count += FLAGS.num_steps
    except tf.errors.OutOfRangeError:
      train_ppl = np.exp(total_loss / word_count)
      print()
      print('ckpt: %s, eval perplexity: %.2f' % (FLAGS.ckpt_path, train_ppl))
      break


if __name__ == '__main__':
  tf.flags.mark_flag_as_required('ckpt_path')
  tf.flags.mark_flag_as_required('filenames')
  tf.flags.mark_flag_as_required('vocab_filenames')

  tf.app.run()

