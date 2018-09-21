import itertools
import collections

import tensorflow as tf

UNK_ID = 0
UNK = '<unk>'


class BaseLanguageModelDataset(object):
  """Base dataset class to be subclassed by `LanguageModelDataset` (training
  and evaluation) and `LanguageModelGeneratorDataset` (generation).
  """
  def __init__(self, char_level, max_vocab_size, min_count, append_period):
    """Constructor.

    Args:
      char_level: bool scalar, language model is at char level (Default) or word
        level.
      max_vocab_size: int scalar, maximum vocabulary size. If > 0, the top 
        `max_vocab_size` most frequent words are kept in vocabulary.
      min_count: int scalar, words whose counts < `min_count` are not included
        in the vocabulary.
      append_period: bool scalar, whether to append '.' to each line of text.
    """
    self._char_level = char_level
    self._max_vocab_size = max_vocab_size
    self._min_count = min_count
    self._append_period = append_period

    self._table_words = None
    self._initializer_iterator = None

  @property
  def iterator_initializer(self):
    return self._iterator_initializer
    
  def build_vocab(self, filenames):
    """Builds vocabulary. 

    '<unk>' is reserved for Unknown token. Has the side effect of setting 
    attribute `table_words` holding the list of vocabulary words.

    Args:
      filenames: list of strings, holding names of text files.
    """
    lines = itertools.chain(*map(open, filenames))
    raw_vocab = collections.Counter()
    for line in lines:
      raw_vocab.update(self._line_to_seq(line))

    raw_vocab = raw_vocab.most_common()
    if self._max_vocab_size > 0:
      raw_vocab = raw_vocab[:self._max_vocab_size]

    self._table_words = [w for w, c in raw_vocab if c >= self._min_count and
        w != '<unk>']
    self._table_words = ['<unk>'] + self._table_words

  def _line_to_seq(self, line):
    """Convert a line of text file into a sequence of tokens.

    Args:
      line: string scalar, a line in a text file ending with '\n'.

    Return:
      seq: a list of string, the sequence of tokens.
    """
    if self._char_level:
      seq = line.strip()
      seq = list(seq + '.' if self._append_period else seq)
    else:
      seq = line.strip().split()
      if self._append_period: seq.append('.')
    return seq


class LanguageModelDataset(BaseLanguageModelDataset):
  """Generates data tensors to be fed to RNN Language Model.

  Call `build_vocab()` to build vocabulary first. Next call `get_tensor_dict()`
  to get tensors `inputs` and `labels` as the input and groundtruth labels for
  training language model.
  """
  def __init__(self,
               char_level=False,
               batch_size=20,
               num_steps=50, 
               max_vocab_size=0,
               min_count=0,
               append_period=True):
    """Constructor.

    Args:
      char_level: bool scalar, language model is at char level (Default) or word
        level.
      batch_size: int scalar, batch size. The tensors `inputs` and `labels` have
        shape [batch_size, num_steps].
      num_steps: int scalar, num of time steps. The tensors `inputs` and `labels`
        have shape [batch_size, num_steps].
      max_vocab_size: int scalar, maximum vocabulary size. If > 0, the top 
        `max_vocab_size` most frequent words are kept in vocabulary.
      min_count: int scalar, words whose counts < `min_count` are not included
        in the vocabulary.
      append_period: bool scalar, whether to append '.' to each line of text. 
    """
    super(LanguageModelDataset, self).__init__(
        char_level, max_vocab_size, min_count, append_period)
    self._batch_size = batch_size
    self._num_steps = num_steps
  
  def get_tensor_dict(self, filenames):
    """Generates tensor dict mapping from tensor names to tensors for training
    and evaluation.

    Args:
      filenames: list of strings, holding names of text files.

    Returns:
      tensor_dict: a dict mapping from tensor names to tensors with shape being:
        inputs: [batch_size, num_steps]
        labels: [batch_size, num_steps]
    """
    table_words = self._table_words
    if table_words is None:
      raise ValueError('`table_words` must be set by calling `build_vocab()`.')

    lines = itertools.chain(*map(open, filenames))
    raw_data = []
    for line in lines:
      raw_data.extend(self._line_to_seq(line))

    table_words = tf.contrib.lookup.index_table_from_tensor(tf.constant(
        table_words), default_value=UNK_ID)
    data = table_words.lookup(tf.constant(raw_data))

    batch_len = tf.floordiv(tf.size(data), self._batch_size)

    num_chunks = tf.floordiv(batch_len - 1, self._num_steps)
    data = tf.to_int32(tf.reshape(data[:self._batch_size*batch_len], 
        [self._batch_size, batch_len]))

    inputs = tf.transpose(tf.reshape(data[:, :num_chunks*self._num_steps],
        [self._batch_size, num_chunks, self._num_steps]), [1, 0, 2])
    labels = tf.transpose(tf.reshape(data[:, 1:num_chunks*self._num_steps+1],
        [self._batch_size, num_chunks, self._num_steps]), [1, 0, 2])

    dataset = tf.data.Dataset.zip((
        tf.data.Dataset.from_tensor_slices(inputs),
        tf.data.Dataset.from_tensor_slices(labels)))

    iterator = dataset.make_initializable_iterator()
    self._iterator_initializer = iterator.initializer
    inputs, labels = iterator.get_next()
    tensor_dict = {'inputs': inputs, 'labels': labels}
    return tensor_dict


class LanguageModelGeneratorDataset(BaseLanguageModelDataset):
  """Generates data tensors to be fed to RNN Language Model for sequence
  generation.

  Call `build_vocab()` to build vocabulary first. Next call `get_tensor_dict()`
  to get tensor `primer` to be used by a sequence generator.
  """  
  def get_tensor_dict(self, filename):
    """Generates tensor dict mapping from tensor names to tensors for sequence
    generation.

    Args:
      filename: string scalar, holding the name of the input text file.

    Returns:
      tensor_dict: a dict mapping from tensor names to tensors with shape being:
        primer: [primer_len]
    """
    table_words = self._table_words
    if table_words is None:
      raise ValueError('`table_words` must be set by calling `build_vocab()`.')

    table_words = tf.contrib.lookup.index_table_from_tensor(tf.constant(
        table_words), default_value=UNK_ID)

    dataset = tf.data.TextLineDataset([filename]) 
    dataset = dataset.map(lambda s: 
        tf.string_split([s], delimiter='' if self._char_level else ' ').values)
    dataset = dataset.map(lambda s: table_words.lookup(s))
    iterator = dataset.make_initializable_iterator()
    self._iterator_initializer = iterator.initializer
    primer = iterator.get_next()
    return {'primer': primer}

  def get_rev_vocab_table(self):
    """Returns the reverse vocabulary table that maps word indices to words."""
    table_words = self._table_words
    if table_words is None:
      raise ValueError('`table_words` must be set by calling `build_vocab()`.')
    rev_vocab_table = tf.contrib.lookup.index_to_string_table_from_tensor(
        self._table_words, default_value=UNK)
    return rev_vocab_table

