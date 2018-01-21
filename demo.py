from language_model import *

train_path = "/home/user/Desktop/lstm_language_model/simple-examples/data/ptb.train.txt"
valid_path = "/home/user/Desktop/lstm_language_model/simple-examples/data/ptb.valid.txt"
test_path = "/home/user/Desktop/lstm_language_model/simple-examples/data/ptb.test.txt"

class WordLevel(object):
  init_scale = 0.1
  init_lr = 1.0
  max_grad_norm = 5.
  num_layers = 2
  num_steps = 20
  embedding_size = 200
  hidden_sizes = (200, 200)
  max_epoch = 4
  max_max_epoch = 13
  keep_prob = 1.0
  lr_decay = 0.5
  batch_size = 20
  vocab_size = 10000

class CharLevel(object):
  init_scale = 0.1
  init_lr = 0.1
  max_grad_norm = 5.
  num_layers = 2
  num_steps = 40
  embedding_size = 100
  hidden_sizes=(100, 100)
  max_epoch = 4
  max_max_epoch = 13
  keep_prob = 1.0
  lr_decay = 0.5
  batch_size = 20
  vocab_size = 50


config = WordLevel()

train_df = DataFeeder(batch_size=config.batch_size, num_steps=config.num_steps, filename=train_path, word_level=True)


lm = RNNLanguageModel( init_scale=config.init_scale,
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
                    vocab_size=config.vocab_size,
                    cell_type="lstm")

lm.train(train_df)

test_df = DataFeeder(batch_size=1, num_steps=1, filename=test_path, word_level=True, vocab=train_df.vocab)
p = lm.score(test_df)

#exclude_ids = set([])
exclude_ids = set([1])
primer = ["the", "meaning", "of", "life", "is"]
#primer = "it seems that"
gen_ids = lm.generate(primer, train_df.vocab, exclude_ids, 100)
gen_words = [train_df.index2item[id_] for id_ in gen_ids]
