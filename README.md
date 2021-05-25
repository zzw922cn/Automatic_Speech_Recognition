# Automatic-Speech-Recognition
End-to-end automatic speech recognition system implemented in TensorFlow.

## Recent Updates
- [x] **Support TensorFlow r1.0** (2017-02-24)
- [x] **Support dropout for dynamic rnn** (2017-03-11)
- [x] **Support running in shell file** (2017-03-11)
- [x] **Support evaluation every several training epoches automatically** (2017-03-11)
- [x] **Fix bugs for character-level automatic speech recognition** (2017-03-14)
- [x] **Improve some function apis for reusable** (2017-03-14)
- [x] **Add scaling for data preprocessing** (2017-03-15)
- [x] **Add reusable support for LibriSpeech training** (2017-03-15)
- [x] **Add simple n-gram model for random generation or statistical use** (2017-03-23)
- [x] **Improve some code for pre-processing and training** (2017-03-23)
- [x] **Replace TABs with blanks and add nist2wav converter script** (2017-04-20)
- [x] **Add some data preparation code** (2017-05-01)
- [x] **Add WSJ corpus standard preprocessing by s5 recipe** (2017-05-05)
- [x] **Restructuring of the project. Updated train.py for usage convinience** (2017-05-06)
- [x] **Finish feature module for timit, libri, wsj, support training for LibriSpeech** (2017-05-14)
- [x] **Remove some unnecessary codes** (2017-07-22)
- [x] **Add DeepSpeech2 implementation code** (2017-07-23)
- [x] **Fix some bugs** (2017-08-06)
- [x] **Add Layer Normalization RNN for efficiency** (2017-08-06)
- [x] **Add Madarian Speech Recognition support** (2017-08-06)
- [x] **Add Capsule Network Model** (2017-12-12)
- [x] **Release 1.0.0 version** (2017-12-14)
- [x] **Add Language Modeling Module** (2017-12-25)
- [x] **Will support TF1.12 soon** (2019-10-17)

## Recommendation
If you want to replace feed dict operation with Tensorflow multi-thread and fifoqueue input pipeline, you can refer to my repo [TensorFlow-Input-Pipeline](https://github.com/zzw922cn/TensorFlow-Input-Pipeline) for more example codes. My own practices prove that fifoqueue input pipeline would improve the training speed in some time.

If you want to look the history of speech recognition, I have collected the significant papers since 1981 in the ASR field. You can read awesome paper list in my repo [awesome-speech-recognition-papers](https://github.com/zzw922cn/awesome-speech-recognition-papers), all download links of papers are provided. I will update it every week to add new papers, including speech recognition, speech synthesis and language modelling. I hope that we won't miss any important papers in speech domain.

All my public repos will be updated in future, thanks for your stars!

## Install and Usage
Currently only python 3.5 is supported.

This project depends on scikit.audiolab, for which you need to have [libsndfile](http://www.mega-nerd.com/libsndfile/) installed in your system.
Clone the repository to your preferred directory and install using:
<pre>
sudo pip3 install -r requirements.txt
sudo python3 setup.py install
</pre>

To use, simply run the following command:
<pre>
python main/timit_train.py [-h] [--mode MODE] [--keep [KEEP]] [--nokeep]
                      [--level LEVEL] [--model MODEL] [--rnncell RNNCELL]
                      [--num_layer NUM_LAYER] [--activation ACTIVATION]
                      [--optimizer OPTIMIZER] [--batch_size BATCH_SIZE]
                      [--num_hidden NUM_HIDDEN] [--num_feature NUM_FEATURE]
                      [--num_classes NUM_CLASSES] [--num_epochs NUM_EPOCHS]
                      [--lr LR] [--dropout_prob DROPOUT_PROB]
                      [--grad_clip GRAD_CLIP] [--datadir DATADIR]
                      [--logdir LOGDIR]

optional arguments:
  -h, --help            show this help message and exit
  --mode MODE           set whether to train or test
  --keep [KEEP]         set whether to restore a model, when test mode, keep
                        should be set to True
  --nokeep
  --level LEVEL         set the task level, phn, cha, or seq2seq, seq2seq will
                        be supported soon
  --model MODEL         set the model to use, DBiRNN, BiRNN, ResNet..
  --rnncell RNNCELL     set the rnncell to use, rnn, gru, lstm...
  --num_layer NUM_LAYER
                        set the layers for rnn
  --activation ACTIVATION
                        set the activation to use, sigmoid, tanh, relu, elu...
  --optimizer OPTIMIZER
                        set the optimizer to use, sgd, adam...
  --batch_size BATCH_SIZE
                        set the batch size
  --num_hidden NUM_HIDDEN
                        set the hidden size of rnn cell
  --num_feature NUM_FEATURE
                        set the size of input feature
  --num_classes NUM_CLASSES
                        set the number of output classes
  --num_epochs NUM_EPOCHS
                        set the number of epochs
  --lr LR               set the learning rate
  --dropout_prob DROPOUT_PROB
                        set probability of dropout
  --grad_clip GRAD_CLIP
                        set the threshold of gradient clipping
  --datadir DATADIR     set the data root directory
  --logdir LOGDIR       set the log directory

</pre>
Instead of configuration in command line, you can also set the arguments above in [timit\_train.py](https://github.com/zzw922cn/Automatic_Speech_Recognition/blob/master/main/timit_train.py) in practice.

Besides, you can also run `main/run.sh` for both training and testing simultaneously! See [run\_timit.sh](https://github.com/zzw922cn/Automatic_Speech_Recognition/blob/master/main/run_timit.sh) for details.

## Performance
### PER based dynamic BLSTM on TIMIT database, with casual tuning because time it limited
![image](https://github.com/zzw922cn/Automatic_Speech_Recognition/blob/master/PER.png)

### LibriSpeech recognition result without LM
**Label**:

it was about noon when captain waverley entered the straggling village or rather hamlet of tully veolan close to which was situated the mansion of the proprietor

**Prediction**:

it was about noon when captain wavraly entered the stragling bilagor of rather hamlent of tulevallon close to which wi situated the mantion of the propriater


**Label**:

the english it is evident had they not been previously assured of receiving the king would never have parted with so considerable a sum and while they weakened themselves by the same measure have strengthened a people with whom they must afterwards have so material an interest to discuss

**Prediction**:

the onglish it is evident had they not being previously showed of receiving the king would never have parted with so considerable a some an quile they weakene themselves by the same measure haf streigth and de people with whom they must afterwards have so material and interest to discuss


**Label**:

one who writes of such an era labours under a troublesome disadvantage

**Prediction**:

one how rights of such an er a labours onder a troubles hom disadvantage


**Label**:

then they started on again and two hours later came in sight of the house of doctor pipt

**Prediction**:

then they started on again and two hours laytor came in sight of the house of doctor pipd


**Label**:

what does he want

**Prediction**:

whit daes he want


**Label**:

there just in front

**Prediction**:

there just infront


**Label**:

under ordinary circumstances the abalone is tough and unpalatable but after the deft manipulation of herbert they are tender and make a fine dish either fried as chowder or a la newberg

**Prediction**:

under ordinary circumstancesi the abl ony is tufgh and unpelitable but after the deftominiculation of hurbourt and they are tender and make a fine dish either fride as choder or alanuburg


**Label**:

by degrees all his happiness all his brilliancy subsided into regret and uneasiness so that his limbs lost their power his arms hung heavily by his sides and his head drooped as though he was stupefied

**Prediction**:

by degrees all his happiness ill his brilliancy subsited inter regret and aneasiness so that his limbs lost their power his arms hung heavily by his sides and his head druped as though he was stupified


**Label**:

i am the one to go after walt if anyone has to i'll go down mister thomas

**Prediction**:

i have the one to go after walt if ety wod hastu i'll go down mister thommas


**Label**:

i had to read it over carefully as the text must be absolutely correct

**Prediction**:

i had to readit over carefully as the tex must be absolutely correct


**Label**:

with a shout the boys dashed pell mell to meet the pack train and falling in behind the slow moving burros urged them on with derisive shouts and sundry resounding slaps on the animals flanks

**Prediction**:

with a shok the boy stash pale mele to meek the pecktrait ane falling in behind the slow lelicg burs ersh tlan with deressive shouts and sudery resounding sleps on the animal slankes


**Label**:

i suppose though it's too early for them then came the explosion

**Prediction**:

i suppouse gho waths two early for them then came the explosion


## Content
This is a powerful library for **automatic speech recognition**, it is implemented in TensorFlow and support training with CPU/GPU. This library contains followings models you can choose to train your own model:
* Data Pre-processing
* Acoustic Modeling
  * RNN
  * BRNN
  * LSTM
  * BLSTM
  * GRU
  * BGRU
  * Dynamic RNN
  * Deep Residual Network
  * Seq2Seq with attention decoder
  * etc.
* CTC Decoding
* Evaluation(Mapping some similar phonemes)  
* Saving or Restoring Model
* Mini-batch Training
* Training with GPU or CPU with TensorFlow
* Keeping logging of epoch time and error rate in disk

## Implementation Details

### Data preprocessing

#### TIMIT corpus

The original TIMIT database contains 6300 utterances, but we find the 'SA' audio files occurs many times, it will lead bad bias for our speech recognition system. Therefore, we removed the all 'SA' files from the original dataset and attain the new TIMIT dataset, which contains only 5040 utterances including 3696 standard training set and 1344 test set.

Automatic Speech Recognition transcribes a raw audio file into character sequences; the preprocessing stage converts a raw audio file into feature vectors of several frames. We first split each audio file into 20ms Hamming windows with an overlap of 10ms, and then calculate the 12 mel frequency ceptral coefficients, appending an energy variable to each frame. This results in a vector of length 13. We then calculate the delta coefficients and delta-delta coefficients, attaining a total of 39 coefficients for each frame. In other words, each audio file is split into frames using the Hamming windows function, and each frame is extracted to a feature vector of length 39 (to attain a feature vector of different length, modify the settings in the file [timit\_preprocess.py](https://github.com/zzw922cn/Automatic-Speech-Recognition/blob/master/src/feature/timit/timit_preprocess.py).

In folder data/mfcc, each file is a feature matrix with size timeLength\*39 of one audio file; in folder data/label, each file is a label vector according to the mfcc file.

If you want to set your own data preprocessing, you can edit [calcmfcc.py](https://github.com/zzw922cn/Automatic-Speech-Recognition/blob/master/src/feature/core/calcmfcc.py) or [timit\_preprocess.py](https://github.com/zzw922cn/Automatic-Speech-Recognition/blob/master/src/feature/timit/timit_preprocess.py).

The original TIMIT dataset contains 61 phonemes, we use 61 phonemes for training and evaluation, but when scoring, we mappd the 61 phonemes into 39 phonemes for better performance. We do this mapping according to the paper [Speaker-independent phone recognition using hidden Markov models](http://repository.cmu.edu/cgi/viewcontent.cgi?article=2768&context=compsci). The mapping details are as follows:

| Original Phoneme(s) | Mapped Phoneme |
| :------------------  | :-------------------: |
| iy | iy |
| ix, ih | ix |
| eh | eh |
| ae | ae |
| ax, ah, ax-h | ax | 
| uw, ux | uw |
| uh | uh |
| ao, aa | ao |
| ey | ey |
| ay | ay |
| oy | oy |
| aw | aw |
| ow | ow |
| er, axr | er |
| l, el | l |
| r | r |
| w | w |
| y | y |
| m, em | m |
| n, en, nx | n |
| ng, eng | ng |
| v | v |
| f | f |
| dh | dh |
| th | th |
| z | z |
| s | s |
| zh, sh | zh |
| jh | jh |
| ch | ch |
| b | b |
| p | p |
| d | d |
| dx | dx |
| t | t |
| g | g |
| k | k |
| hh, hv | hh |
| bcl, pcl, dcl, tcl, gcl, kcl, q, epi, pau, h# | h# |
 

#### LibriSpeech corpus

LibriSpeech is a corpus of approximately 1000 hours of 16kHz read English speech. It can be downloaded from [here](http://www.openslr.org/12/)

In order to preprocess LibriSpeech data, download the dataset from the above mentioned link, extract it and run the following:
<pre>
cd feature/libri
python libri_preprocess.py -h 
usage: libri_preprocess [-h]
                        [-n {dev-clean,dev-other,test-clean,test-other,train-clean-100,train-clean-360,train-other-500}]
                        [-m {mfcc,fbank}] [--featlen FEATLEN] [-s]
                        [-wl WINLEN] [-ws WINSTEP]
                        path save

Script to preprocess libri data

positional arguments:
  path                  Directory of LibriSpeech dataset
  save                  Directory where preprocessed arrays are to be saved

optional arguments:
  -h, --help            show this help message and exit
  -n {dev-clean,dev-other,test-clean,test-other,train-clean-100,train-clean-360,train-other-500}, --name {dev-clean,dev-other,test-clean,test-other,train-clean-100,train-clean-360,train-other-500}
                        Name of the dataset
  -m {mfcc,fbank}, --mode {mfcc,fbank}
                        Mode
  --featlen FEATLEN     Features length
  -s, --seq2seq         set this flag to use seq2seq
  -wl WINLEN, --winlen WINLEN
                        specify the window length of feature
  -ws WINSTEP, --winstep WINSTEP
                        specify the window step length of feature
</pre>

The processed data will be saved in the "save" path. 

To train the model, run the following:
<pre>
python main/libri_train.py -h 
usage: libri_train.py [-h] [--task TASK] [--train_dataset TRAIN_DATASET]
                      [--dev_dataset DEV_DATASET]
                      [--test_dataset TEST_DATASET] [--mode MODE]
                      [--keep [KEEP]] [--nokeep] [--level LEVEL]
                      [--model MODEL] [--rnncell RNNCELL]
                      [--num_layer NUM_LAYER] [--activation ACTIVATION]
                      [--optimizer OPTIMIZER] [--batch_size BATCH_SIZE]
                      [--num_hidden NUM_HIDDEN] [--num_feature NUM_FEATURE]
                      [--num_classes NUM_CLASSES] [--num_epochs NUM_EPOCHS]
                      [--lr LR] [--dropout_prob DROPOUT_PROB]
                      [--grad_clip GRAD_CLIP] [--datadir DATADIR]
                      [--logdir LOGDIR]

optional arguments:
  -h, --help            show this help message and exit
  --task TASK           set task name of this program
  --train_dataset TRAIN_DATASET
                        set the training dataset
  --dev_dataset DEV_DATASET
                        set the development dataset
  --test_dataset TEST_DATASET
                        set the test dataset
  --mode MODE           set whether to train, dev or test
  --keep [KEEP]         set whether to restore a model, when test mode, keep
                        should be set to True
  --nokeep
  --level LEVEL         set the task level, phn, cha, or seq2seq, seq2seq will
                        be supported soon
  --model MODEL         set the model to use, DBiRNN, BiRNN, ResNet..
  --rnncell RNNCELL     set the rnncell to use, rnn, gru, lstm...
  --num_layer NUM_LAYER
                        set the layers for rnn
  --activation ACTIVATION
                        set the activation to use, sigmoid, tanh, relu, elu...
  --optimizer OPTIMIZER
                        set the optimizer to use, sgd, adam...
  --batch_size BATCH_SIZE
                        set the batch size
  --num_hidden NUM_HIDDEN
                        set the hidden size of rnn cell
  --num_feature NUM_FEATURE
                        set the size of input feature
  --num_classes NUM_CLASSES
                        set the number of output classes
  --num_epochs NUM_EPOCHS
                        set the number of epochs
  --lr LR               set the learning rate
  --dropout_prob DROPOUT_PROB
                        set probability of dropout
  --grad_clip GRAD_CLIP
                        set the threshold of gradient clipping, -1 denotes no
                        clipping
  --datadir DATADIR     set the data root directory
  --logdir LOGDIR       set the log directory
</pre>

where the "datadir" is the "save" path used in preprocess stage.

#### Wall Street Journal corpus

TODO

### Core Features
+ dynamic RNN(GRU, LSTM)
+ Residual Network(Deep CNN)
+ CTC Decoding
+ TIMIT Phoneme Edit Distance(PER)

## Future Work
- [ ] Release pretrained English ASR model
- [ ] Add Attention Mechanism
- [ ] Add Speaker Verification
- [ ] Add TTS

## License
MIT

## Contact Us
If this program is helpful to you, please give us a **star or fork** to encourage us to keep updating. Thank you! Besides, any issues or pulls are appreciated.

Collaborators: 

[zzw922cn](https://github.com/zzw922cn)

[deepxuexi](https://github.com/deepxuexi)

[hiteshpaul](https://github.com/hiteshpaul)

[xxxxyzt](https://github.com/xxxxyzt)
