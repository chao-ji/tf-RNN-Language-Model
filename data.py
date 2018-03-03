from __future__ import division

import codecs
import collections

import tensorflow as tf

UNK_ID = 0
UNK = "<unk>"


class LanguageModelDataset(object):
  def __init__(self, hparams):
    vocab_elements = _read_raw_vocab_elements(
        hparams.file_vocab, hparams.word_level)

    self._vocab, _ = _build_vocab(
        hparams.vocab_size, vocab_elements)
    (self._initializer, self._in_ids, self._out_ids, self._batch_size
          ) = self._get_iterator(hparams)

  @property
  def in_ids(self):
    return self._in_ids

  @property
  def out_ids(self):
    return self._out_ids

  @property
  def batch_size(self):
    return self._batch_size

  @property
  def vocab(self):
    return self._vocab

  def init_iterator(self, sess):
    sess.run(self._initializer)

  def _get_iterator(self, hparams):
    elements = _read_raw_vocab_elements(
        hparams.file_data, hparams.word_level)
    elements = elements if hparams.word_level else list(elements)
    ids = self.vocab.lookup(tf.constant(elements))

    batch_len = tf.floordiv(
        tf.size(ids), hparams.batch_size, name="batch_length")
    num_epochs = tf.floordiv(
        (batch_len - 1), hparams.num_steps, name="num_epochs")

    ids = tf.reshape(ids[:hparams.batch_size*batch_len],
        [hparams.batch_size, batch_len])

    x = tf.transpose(tf.reshape(ids[:, :num_epochs*hparams.num_steps],
        [hparams.batch_size, num_epochs, hparams.num_steps]), [1, 0, 2])
    y = tf.transpose(tf.reshape(ids[:, 1:num_epochs*hparams.num_steps+1],
        [hparams.batch_size, num_epochs, hparams.num_steps]), [1, 0, 2])

    dataset_x = tf.data.Dataset.from_tensor_slices(x)
    dataset_y = tf.data.Dataset.from_tensor_slices(y)

    dataset = tf.data.Dataset.zip((dataset_x, dataset_y))
    dataset = dataset.map(
        lambda x, y: (x, y, tf.shape(x)[0]))

    iterator = dataset.make_initializable_iterator()
    in_ids, out_ids, batch_size = iterator.get_next()

    return iterator.initializer, in_ids, out_ids, batch_size


class GeneratorInput(object):
  def __init__(self, hparams):
    vocab_elements = _read_raw_vocab_elements(
        hparams.file_vocab, hparams.word_level)
    self._vocab, self._reverse_vocab = _build_vocab(
        hparams.vocab_size, vocab_elements)

    self._input_text = tf.placeholder(tf.string, shape=[])
    delimiter = " " if hparams.word_level else ""
    self._in_ids = self.vocab.lookup(tf.expand_dims(
        tf.string_split([self._input_text], delimiter=delimiter).values, 0))

    self._batch_size = tf.constant(1)

  @property
  def in_ids(self):
    return self._in_ids

  @property
  def batch_size(self):
    return self._batch_size

  @property
  def input_text(self):
    return self._input_text

  @property
  def vocab(self):
    return self._vocab

  @property
  def reverse_vocab(self):
    return self._reverse_vocab

def _read_raw_vocab_elements(filename, word_level):
  with codecs.getreader("utf-8")(
      tf.gfile.GFile(filename, mode="rb")) as f:
    raw_vocab_elements = f.read() # unicode string
    if word_level:
      raw_vocab_elements = raw_vocab_elements.replace("\n", "<eos>").split()
  return raw_vocab_elements


def _build_vocab(vocab_size, elements):
  counter = collections.Counter(elements)
  vocab = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
  vocab = zip(*vocab[:vocab_size])[0]

  return tf.contrib.lookup.index_table_from_tensor(
      tf.constant(vocab, dtype=tf.string), default_value=UNK_ID), list(vocab)
