from __future__ import division

import codecs
import collections

import tensorflow as tf

UNK_ID = 0


class SequenceDataset(object):
  def __init__(self, hparams):
    (self._initializer, self._in_ids, self._out_ids
        ) = self._get_iterator(hparams)

  @property
  def in_ids(self):
    return self._in_ids

  @property
  def out_ids(self):
    return self._out_ids

  @property
  def vocab(self):
    return self._vocab

  def init_iterator(self, sess):
    sess.run(self._initializer)

  def _read_raw_vocab_elements(self, hparams):
    filename = hparams.full_path
    with codecs.getreader("utf-8")(
        tf.gfile.GFile(filename, mode="rb")) as f:
      raw_vocab_elements = f.read() # unicode string
      if hparams.word_level:
        raw_vocab_elements = raw_vocab_elements.replace("\n", "<eos>").split()
    return raw_vocab_elements

  def _build_vocab(self, vocab_size, elements):
    counter = collections.Counter(elements)
    vocab = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    vocab = vocab[:vocab_size]
    vocab = tf.contrib.lookup.index_table_from_tensor(
        tf.constant(zip(*vocab)[0], dtype=tf.string), default_value=UNK_ID)
    return vocab

  def _get_iterator(self, hparams):
    elements = self._read_raw_vocab_elements(hparams)
    self._vocab = vocab = self._build_vocab(hparams.vocab_size, elements)
    ids = vocab.lookup(tf.constant(elements))

    batch_len = tf.floordiv(
        tf.size(ids), hparams.batch_size, name="batch_length")   
    num_epochs = tf.floordiv(
        (batch_len - 1), hparams.num_steps, name="num_epochs")

    ids = tf.reshape(ids[:hparams.batch_size*batch_len],
        [hparams.batch_size, batch_len])

    x = tf.transpose(tf.reshape(ids[:, :num_epochs*hparams.num_steps],
        [-1, num_epochs, hparams.num_steps]), [1, 0, 2])
    y = tf.transpose(tf.reshape(ids[:, 1:num_epochs*hparams.num_steps+1],
        [-1, num_epochs, hparams.num_steps]), [1, 0, 2])
   
    dataset_x = tf.data.Dataset.from_tensor_slices(x)
    dataset_y = tf.data.Dataset.from_tensor_slices(y)

    dataset = tf.data.Dataset.zip((dataset_x, dataset_y))
    
    iterator = dataset.make_initializable_iterator()
    in_ids, out_ids = iterator.get_next()

    return iterator.initializer, in_ids, out_ids
