# Automatic-Speech-Recognition
End-to-end automatic speech recognition system implemented in TensorFlow.

## Recent Updates
- [x] Support TensorFlow r1.0(2017-02-24)

## Content
This is a powerful library for automatic speech recognition, it is implemented in TensorFlow and support training with CPU/GPU. This library contains followings models you can choose to train your own model:
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

## Usage
<pre>
python train.py
	--mfcc_dir '/data/mfcc/'
	--label_dir '/data/label/'
	--keep False
	--save True
	--evaluation False
	--learning_rate 0.001
	--batch_size 32
	--num_feature 39
	--num_hidden 128
	--num_classes 28
	--save_dir '/src/save/'
	--restore_from '/src/save/'
	--model_checkpoint_path '/src/save/'
</pre>
Instead of configuration in command line, you can also set the arguments above in [train.py](https://github.com/zzw922cn/Automatic-Speech-Recognition/blob/master/src/main/train.py) in practice.

## Implementation Details

### Data preprocessing
The original TIMIT database contains 6300 utterances, but we find the 'SA' audio files occurs many times, it will lead bad bias for our speech recognition system. Therefore, we removed the all 'SA' files from the original dataset and attain the new TIMIT dataset, which contains only 5040 utterances including 3696 standard training set and 1344 test set.

Automatic Speech Recognition is to transcribe a raw audio file into character sequences. Data preprocessing is to convert a raw audio file into feature vectors of several frames. Here, we first split each audio file by a 20ms hamming window with an overlap of 10ms, and then calculate the 12 mel frequency ceptral coefficients appended by an energy variable for each frame. Based on this vector of length 13, we calculate the delta coefficients and delta-delta coefficients, therefore, we attain totally 39 coefficients for each frame. Therefore, each audio file is splited to several frames by hamming window, and each frame is extracted to a feature vector of length 39. If you want to attain the feature vector of different length, you can reset the settings in the file [timit_preprocess.py](https://github.com/zzw922cn/Automatic-Speech-Recognition/blob/master/src/feature/timit_preprocess.py).

In folder data/mfcc, each file is a feature matrix with size timeLength*39 of one audio file; in folder data/label, each file is a label vector according to the mfcc file.

If you want to set your own data preprocessing, you can edit [calcmfcc.py](https://github.com/zzw922cn/Automatic-Speech-Recognition/blob/master/src/feature/calcmfcc.py) or [timit_preprocess.py](https://github.com/zzw922cn/Automatic-Speech-Recognition/blob/master/src/feature/timit_preprocess.py).

Since the original TIMIT dataset contains 61 phonemes, we use 61 phonemes for training and evaluation, but when scoring, we mappd the 61 phonemes into 39 phonemes for better performance. We do this mapping according to the paper [Speaker-independent phone recognition using hidden Markov models](http://repository.cmu.edu/cgi/viewcontent.cgi?article=2768&context=compsci). The mapping details are as follows:

| original phoneme(s) | mapped into phoneme |
| :------------------  | :-------------------: |
| ux | uw |
| axr | er |
| em | m |
| nx, n  | en |
| eng | ng |
| hv | hh |
| cl, bcl, dcl, gcl, epi, h#, kcl, pau, pcl, tcl, vcl | sil |
| l | el |
| zh | sh |
| aa | ao |
| ix | ih |
| ax | ah | 
 

### Core Features
+ dynamic RNN(GRU, LSTM)
+ Residual Network(Deep CNN)
+ CTC Decoding
+ TIMIT Phoneme Edit Distance(PER)

### Future Work
- [ ] Add Attention Mechanism
- [ ] Add more efficient dynamic computation graph without padding
- [ ] List experimental results 
- [ ] Implement more ASR models following newest investigations 


