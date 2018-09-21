r"""Executable for training language model.

Example:
  python run_training.py \
      --filenames=/PATH/TO/FILENAMES \
      --vocab_filenames=/PATH/TO/VOCAB_FILENAMES \ 
      --ckpt_path=/PATH/TO/CKPT

Checkpoint file paths will have prefix `/PATH/TO/CKPT`. 
"""
import tensorflow as tf
import numpy as np

from dataset import LanguageModelDataset
from prediction_model import RNNLanguagePredictionModel
from model_runners import RNNLanguageModelTrainer
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
flags.DEFINE_float('keep_prob', 0.5, 'Dropout rate equals `1 - keep_prob`.') 
flags.DEFINE_integer('num_layers', 2, 'Num of RNN layers.')
flags.DEFINE_float('init_scale', 0.05, 'The random uniform initializer has '
    'range [-init_scale, init_scale].')

flags.DEFINE_float('init_lr', 1.0, 'Initial learning rate.')
flags.DEFINE_float('lr_decay', 0.8, 'Learning rate is decayed by a factor of ' 
    '`lr_decay`.')
flags.DEFINE_float('max_grad_norm', 5.0, 'Maximum gradient norm.')
flags.DEFINE_integer('num_epochs_no_decay', 6, 'The learning rate remains for '
    'the first `num_epochs_no_decay` epochs.')
flags.DEFINE_integer('num_epochs', 39, 'Num of epochs to train.')

flags.DEFINE_string('ckpt_path', '/tmp/rnn/model.ckpt', 'Prefix of the paths to '
    'the checkpoint files.')
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
      keep_prob=FLAGS.keep_prob,
      num_layers=FLAGS.num_layers,
      init_scale=FLAGS.init_scale,
      is_training=True)

  trainer = RNNLanguageModelTrainer(
      prediction_model=prediction_model,
      batch_size=FLAGS.batch_size,
      init_lr=FLAGS.init_lr,
      lr_decay=FLAGS.lr_decay,
      max_grad_norm=FLAGS.max_grad_norm,
      num_epochs_no_decay=FLAGS.num_epochs_no_decay)

  to_be_run_dict = trainer.train(FLAGS.filenames, dataset)
  saver = tf.train.Saver(max_to_keep=None)

  sess = tf.InteractiveSession()
  sess.run(tf.tables_initializer())
  sess.run(tf.global_variables_initializer())

  for i in range(FLAGS.num_epochs):
    sess.run(dataset.iterator_initializer)
    state_val = eval_init_state(trainer, sess)

    total_loss = 0 
    word_count = 0

    while True:
      try:
        feed_dict = get_init_state_to_value_feed_dict(trainer, state_val)
        result_dict = sess.run(to_be_run_dict, feed_dict=feed_dict)
        state_val = result_dict['final_state']

        total_loss += result_dict['loss']
        word_count += FLAGS.num_steps
      except tf.errors.OutOfRangeError:
        train_ppl = np.exp(total_loss / word_count)
        print('epoch: %d, train perplexity: %.2f, learning rate: %f' % 
            (i, train_ppl, result_dict['learning_rate']))
        break

    trainer.increment_epoch_index(sess)
    saver.save(sess, FLAGS.ckpt_path, global_step=i)


if __name__ == '__main__':
  tf.flags.mark_flag_as_required('filenames')
  tf.flags.mark_flag_as_required('vocab_filenames')

  tf.app.run()
