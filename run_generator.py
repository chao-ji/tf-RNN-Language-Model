r"""Executable for sequence generation.

Example:
  python run_generator.py \
    --primer_file/PATH/TO/PRIMER_FILE 
    --vocab_filenames=/PATH/TO/VOCAB_FILENAMES
    --ckpt_path=/PATH/TO/CKPT

"""
import tensorflow as tf
import numpy as np

from dataset import LanguageModelGeneratorDataset
from prediction_model import RNNLanguagePredictionModel
from model_runners import RNNLanguageModelSeqGenerator
from model_runners import eval_init_state
from model_runners import get_init_state_to_value_feed_dict

flags = tf.app.flags


flags.DEFINE_boolean('char_level', False, 'Language model is at char level '
    '(Default) or word level.')
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

flags.DEFINE_integer('gen_seq_len', 100, 'The length of sequence to be '
    'generated.')
flags.DEFINE_boolean('use_sampling', True, 'Whether to use sampling strategy '
    '(Default) or argmax strategy.')

flags.DEFINE_string('ckpt_path', None, 'Path to the checkpoint'
    ' file to be loaded.')
flags.DEFINE_list('vocab_filenames', None, 'Comma-separated list of names of '
    'text files to build vocabulary from.')
flags.DEFINE_string('primer_file', None, 'Path to the text file containing '
    'primer sequences. Each line holds a single primer sequence.')

FLAGS = flags.FLAGS


def main(_):
  dataset = LanguageModelGeneratorDataset(char_level=FLAGS.char_level,
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

  generator = RNNLanguageModelSeqGenerator(
      prediction_model=prediction_model, 
      gen_seq_len=FLAGS.gen_seq_len, 
      use_sampling=FLAGS.use_sampling)


  to_be_run_dict = generator._generate_basic(FLAGS.primer_file, dataset)
  saver = tf.train.Saver()

  sess = tf.InteractiveSession()
  sess.run(tf.tables_initializer())
  sess.run(tf.global_variables_initializer())

  delimiter = b'' if FLAGS.char_level else b' '

  sess.run(dataset.iterator_initializer)
  saver.restore(sess, FLAGS.ckpt_path)

  while True:
    try:
      gen_seq = sess.run(to_be_run_dict['gen_seq'])
      seq = delimiter.join(list(gen_seq)).decode('utf-8')
      print(seq, '\n')
    except tf.errors.OutOfRangeError:
      break


if __name__ == '__main__':
  tf.flags.mark_flag_as_required('ckpt_path')
  tf.flags.mark_flag_as_required('primer_file')
  tf.flags.mark_flag_as_required('vocab_filenames')

  tf.app.run()

