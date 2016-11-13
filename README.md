# Automatic-Speech-Recognitio
End-to-end character-level automatic speech recognition system implemented in TensorFlow.

## Content
Automatic Speech Recognition implemented in TensorFlow contains followings:
* Data Pre-processing
* Acoustic Modeling(RNN,LSTM,BRNN,BLSTM,etc.)
* CTC Decoding
* Evaluation  
* Saving or Restoring Model
* Mini-batch Training
* Training with GPU or CPU with TensorFlow

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
Automatic Speech Recognition is to transcribe a raw audio file into character sequences. Data preprocessing is to convert a raw audio file into feature vectors of several frames. Here, we first split each audio file by a 20ms hamming window with no overlap, and then calculate the 12 mel frequency ceptral coefficients appended by a energy variable for each frame. Based on this vector of length 13, we calculate the delta coefficients and delta-delta coefficients, totally 39 coefficients for each frame. Therefore, each audio file is splited to several frames by hamming window, and each frame is extracted to a feature vector of length 39.

In folder data/mfcc, each file is a feature matrix with size timeLength*39 of one audio file; in folder data/label, each file is a label vector according to the mfcc file.

If you want to set your own data preprocessing, you can edit [calcmfcc.py](https://github.com/zzw922cn/Automatic-Speech-Recognition/blob/master/src/feature/calcmfcc.py) or [preprocess.py](https://github.com/zzw922cn/Automatic-Speech-Recognition/blob/master/src/feature/preprocess.py).

### Acoustic Model
TODO

### CTC Decoding
TODO

### Evaluation
TODO

## Future work
TODO

