from lstm import *

train_path = "/home/user/Desktop/lstm_language_model/simple-examples/data/ptb.train.txt"
valid_path = "/home/user/Desktop/lstm_language_model/simple-examples/data/ptb.valid.txt"
test_path = "/home/user/Desktop/lstm_language_model/simple-examples/data/ptb.test.txt"


class Small(object):
  init_scale = 0.1
  init_lr = 1.0
  max_grad_norm = 5.
  num_layers = 2
  num_steps = 20
  embedding_size = 200
  hidden_sizes=(200, 200)
  max_epoch = 4
  max_max_epoch = 13
  keep_prob = 1.0 
  lr_decay = 0.5
  batch_size = 20
  vocab_size = 10000

train_config = Small()

test_config = Small()
test_config.batch_size = 1
test_config.num_steps = 1 
test_config.keep_prob = 1.0

train_df = DataFeeder(batch_size=train_config.batch_size, num_steps=train_config.num_steps, filename=train_path)

initializer = tf.random_uniform_initializer(-train_config.init_scale,
                                             train_config.init_scale)

sess = tf.InteractiveSession()

#with tf.variable_scope("Model", reuse=None, initializer=initializer):
lm = LanguageModel( init_scale=train_config.init_scale,
                    init_lr=train_config.init_lr,
                    max_grad_norm=train_config.max_grad_norm,
                    num_layers=train_config.num_layers,
                    num_steps=train_config.num_steps,
                    embedding_size=train_config.embedding_size,
                    hidden_sizes=train_config.hidden_sizes,
                    max_epoch=train_config.max_epoch,
                    max_max_epoch=train_config.max_max_epoch,
                    keep_prob=train_config.keep_prob,
                    lr_decay=train_config.lr_decay,
                    batch_size=train_config.batch_size,
                    vocab_size=train_config.vocab_size)

lm.train(train_df, sess)

test_df = DataFeeder(batch_size=1, num_steps=1, filename=test_path, vocab=train_df.vocab)

p = lm.score(test_df, sess)
