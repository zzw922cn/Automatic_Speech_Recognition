# Automatic-Speech-Recognition
Character-level end-to-end automatic speech recognition in Tensorflow.

===========================
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

### Acoustic Model

### CTC Decoding

### Evaluation

## Future work
