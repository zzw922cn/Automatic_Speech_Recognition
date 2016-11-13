# Automatic-Speech-Recognition
Character-level end-to-end automatic speech recognition in Tensorflow.

==============================

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
You can also set the arguments above in /src/main/train.py file.

## Implementation Details

### Data preprocessing
Automatic Speech Recognition is to transcribe a raw audio file into character sequences. Data preprocessing is to convert a raw audio file into feature vectors of several frames. Here, we first split each audio file by a 20ms hamming window with no overlap, and then calculate the 12 mel frequency ceptral coefficients appended by a energy variable for each frame. Based on this vector of length 13, we calculate the delta coefficients and delta-delta coefficients, totally 39 coefficients for each frame. Therefore, each audio file is splited to several frames by hamming window, and each frame is extracted to a feature vector of length 39.

In folder data/mfcc, each file is a feature matrix with size timeLength*39 of one audio file; in folder data/label, each file is a label vector according to the mfcc file.

If you want to set your own data preprocessing, you can edit [calcmfcc.py](https://github.com/zzw922cn/Automatic-Speech-Recognition/blob/master/src/feature/calcmfcc.py) or [data_pre_ch_for_libri.py](https://github.com/zzw922cn/Automatic-Speech-Recognition/blob/master/src/feature/data_pre_ch_for_libri.py).

### Acoustic Model

### CTC Decoding

### Evaluation

## Future work

