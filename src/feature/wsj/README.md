# WSJ corpus

## preparation
You should use kaldi to preprocess the WSJ corpus, since original WSJ is several *.zip files, you should first unzip them one by one.
Then, you should carefully look up the **wsj0.link.log** and **wsj1.link.log**, since it may miss some lines by index. Therefore, you should insert some lines following the directory id.

According to the two log files, you can rename the CD directory by their corresponding new name.

Then you can execute the kaldi/egs/wsj/s5/local/wsj\_data\_prep.sh, the command may be like:
`
./local/wsj_data_prep.sh /media/pony/DLdigest/study/ASR/corpus/wsj/wsj0/??-{?,??}.? /media/pony/DLdigest/study/ASR/corpus/wsj/wsj1/??-{?,??}.?
`

If you see `Data preparation succeeded`, the preparation of WSJ corpus has been finished. You can find the file list in the directory `kaldi/egs/wsj/s5/data/local/data`. 

Wish you succeed!
