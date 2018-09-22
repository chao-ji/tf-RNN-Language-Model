### RNN Language Model

A TensorFlow implementation of RNN language model. Three executables are provided for training (`run_training.py`), evaluation (`run_evaluation.py`) and sequence generation (`run_generator.py`).

### Usage
Get a copy of the repository:
```
git clone git@github.com:chao-ji/tf-RNN-Language-Model.git
```

The input data for `run_training.py`, `run_evaluation.py` or `run_generator.py` are one or more text files, which are interpreted as a sequence of tokens. Depending on the flag `--char_level`, the tokens are either individual bytes, for example, ASCII charachers (`--char_level=True`, ), or multi-byte sequences, for example, English words or multi-byte unicode characters (`--char_level=False`). Note that in the latter case, each multi-byte sequence must not contain the space character, and the input files are supposed to contain space-delimited multi-byte sequences (e.g. `multi_byte_seq1[SPACE]multi_byte_seq2 ...`)

##### Vocabulary
A vocabulary is needed to map words to indices. You must provide text file(s) on which the vocabulary will be built (`--vocab_filenames=`). Note that the files used to generate vocabulary do not have to be the same as the input text files (specified by `--filenames`).


##### Training and Evaluation
For training and evaluation, the sequence of bytes in each of the input text files (`--filenames`) are concatenated into a single sequence, and then cut into `--batch_size` continuous sequences, each of which is further cut into segments of length `--num_steps`. So each mini-batch holds a matrix of word indices of shape `[batch_size, num_steps]`.

To perform training, run
```
  python run_training.py \
      --filenames=/PATH/TO/FILENAMES \
      --vocab_filenames=/PATH/TO/VOCAB_FILENAMES \ 
      --ckpt_path=/PATH/TO/CKPT
```
A checkpoint file will be saved at the end of each epoch (the input files are iterated once in one epoch), and they will have prefix `/PATH/TO/CKPT`.

To perform evaluation, run

```
  python run_evaluation.py \
      --filenames=/PATH/TO/FILENAMES \
      --vocab_filenames=/PATH/TO/VOCAB_FILENAMES \ 
      --ckpt_path=/PATH/TO/CKPT_FILE
```
`/PATH/TO/CKPT_FILE` is the path to the checkpoint file to be evaluated (e.g. `/path/to/model.ckpt-100`).


##### Sequence Generation

To generate sequence using a trained language model, you need to provide a *primer* sequence to be read by the language model, so that it can generate new tokens one at a time based on those generated in the past.

The primer sequences should be saved in a text file, where each line holds a single primer sequence.

To perform sequence generation, run

```
  python run_generator.py \
    --primer_file=/PATH/TO/PRIMER_FILE \
    --vocab_filenames=/PATH/TO/VOCAB_FILENAMES \
    --ckpt_path=/PATH/TO/CKPT_FILE
```

### Example of Sequence Generation

Here is an example of generating sequences using a trained language model.

A character-level language model (`--char_level=True`) is trained using default parameter settings on [PennTreeBank dataset](https://catalog.ldc.upenn.edu/ldc99t42). Sequence generation is primed at `the housing market has` (22 characters), and is set to generate 100 additional characters (`--gen_seq_len=100`)

We load the checkpoint files at different stages of training, and print out the sequences generated after `the housing market has`.

0 steps
```
z'1*-Ne2&$qkhfk64ji*h 0hrm42rxch5k djau jj8z8*znN<de56pi/ejh1//virin\8\-w*749&/q#ebopi>> 1g11v3x#v01
```
200 steps
```
 t  ec<.crdnlf ti tjl toi.   rnn fgfoi r nlfortm n criaaogieuetogt<harda h  c.tnkcui alogos r o an m
```
400 steps
```
em liun an> lin oon l lrutyrs hipr bte' ron sped <fo mo glbapd rtpee vfg focteocum che mte fata inra
```
600 steps 
```
 moru <unkukice folint uingcd at is thu pusecqon* ons on lise bos fere on N N prile susteed lood fim
```
800 steps
```
 hocewt ltopirtae voy apsut.mlineirnig ank N thet 'roung and dercon way yapt slont cobmen protipion 
```
1000 steps 
```
 for ofseres saderes forengtame poncrams to neverhing of.trapes <unk>.ther in to my.ughary salkaded 
```
2000 steps 
```
 <unk> ge.lusterdal in estorpant-bunecure.shings aderhaders after most aid <unk> cecfisions decarss 
```
3000 steps  
```
 robinal N.the from part N N.new upary action in increase done N up N <unk> plemt million indearca d
```
4000 steps 
```
 on the congressio 'll N N on the company flag ailings aver plungers for a third firstment improvide
```
5000 steps
```
 been attacked it government goods ocpation officials said.the shares from N N for periine hasweredd
```
A few epochs
```
 said one strate-year N.but when the convertible problems products think in the european level.altho
```

Initialized with random weights (0 steps), the model seems to be generating completely random characters. With just a few hundred steps of training, the model starts to learn to insert spaces between "words", and also learns how to put vowels and consonants so that the "words" are pronounceable. After about 3000 steps, the model starts to spit out words with generally corret spelling. Finally, after a few epochs of training, even the grammar starts to make sense -- e.g. `said` follows `has` in the primer sequence.


