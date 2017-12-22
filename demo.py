from lstm import *

train_path = "/home/user/Desktop/lstm_language_model/simple-examples/data/ptb.train.txt"
test_path = "/home/user/Desktop/lstm_language_model/simple-examples/data/ptb.test.txt"
valid_path = "/home/user/Desktop/lstm_language_model/simple-examples/data/ptb.valid.txt"


class Small(object):
  init_scale = 0.1
  init_lr = 1.0
  max_grad_norm = 5.
  num_layers = 2
  num_steps = 20
  embedding_size = 200
  max_epoch = 4
  max_max_epoch = 13
  keep_prob = 1.0 
  lr_decay = 0.5
  batch_size = 20
  vocab_size = 10000

class Medium(object):
  init_scale = 0.05
  init_lr = 1.0
  max_grad_norm = 5.
  num_layers = 2
  num_steps = 35
  embedding_size = 650
  max_epoch = 6
  max_max_epoch = 39
  keep_prob = 0.5
  lr_decay = 0.8
  batch_size = 20
  vocab_size = 10000


train_config = Small()

test_config = Small()
test_config.batch_size = 1
test_config.num_steps = 1
test_config.keep_prob = 1.0

train_df = DataFeeder(filename=train_path, config=train_config)
test_df = DataFeeder(filename=test_path, config=test_config, vocab=train_df.vocab)

initializer = tf.random_uniform_initializer(-train_config.init_scale,
                                             train_config.init_scale)

sess = tf.InteractiveSession()
with tf.variable_scope("Model", reuse=None, initializer=initializer):
  lm = LanguageModel(True, train_df, config=train_config)
  lm.train(train_df, sess)

tvars = tf.trainable_variables()
z = sess.run(tvars)


with tf.variable_scope("Model", reuse=True):
  lm_test = LanguageModel(False, test_df, config=test_config)
  p = lm_test.score(test_df, sess)
 
