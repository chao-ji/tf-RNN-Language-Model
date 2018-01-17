from lstm import *

train_path = "/home/chaoji/Desktop/lstm_language_model/simple-examples/data/ptb.train.txt"
valid_path = "/home/chaoji/Desktop/lstm_language_model/simple-examples/data/ptb.valid.txt"
test_path = "/home/chaoji/Desktop/lstm_language_model/simple-examples/data/ptb.test.txt"


class Small(object):
  init_scale = 0.1
  init_lr = 1.0
  max_grad_norm = 5.
  num_layers = 2
  num_steps = 20
  embedding_size = 200
  hidden_sizes=(200, 200)
  max_epoch = 4
  max_max_epoch = 1
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
  hidden_sizes=(650, 650)
  max_epoch = 6
  max_max_epoch = 39
  keep_prob = 0.5
  lr_decay = 0.8
  batch_size = 20
  vocab_size = 10000


config = Small()


train_df = DataFeeder(batch_size=config.batch_size, num_steps=config.num_steps, filename=train_path)
g = train_df.batch_generator()
g.next()

sess = tf.InteractiveSession()

lm = LanguageModel( init_scale=config.init_scale,
                    init_lr=config.init_lr,
                    max_grad_norm=config.max_grad_norm,
                    num_layers=config.num_layers,
                    num_steps=config.num_steps,
                    embedding_size=config.embedding_size,
                    hidden_sizes=config.hidden_sizes,
                    max_epoch=config.max_epoch,
                    max_max_epoch=config.max_max_epoch,
                    keep_prob=config.keep_prob,
                    lr_decay=config.lr_decay,
                    batch_size=config.batch_size,
                    vocab_size=config.vocab_size)

lm.train(train_df, sess)

#test_df = DataFeeder(batch_size=1, num_steps=1, filename=test_path, vocab=train_df.vocab)

#p = lm.score(test_df, sess)

index2word = zip(*sorted([(c, w) for w, c in train_df.vocab.items()], key=lambda p: p[0]))[1]

exclude_ids = set([1])
seed_word_list = ["the", "meaning", "of", "life", "is"]
gen_words = lm.generate(seed_word_list, train_df.vocab, index2word, exclude_ids, 100, sess)
