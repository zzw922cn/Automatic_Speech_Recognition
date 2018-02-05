# encoding: utf-8
# ******************************************************
# Author       : zzw922cn
# Last modified: 2017-12-09 11:00
# Email        : zzw922cn@gmail.com
# Filename     : madarian_preprocess.py
# Description  : Feature preprocessing for some Madarian dataset
# ******************************************************

from speechvalley.feature.madarian import convertDigit2Character
from speechvalley.feature.madarian import convertCharacter2Digit

class DigitPrecessor(object):

    def __init__(self, mode):
        assert mode=='digit2char' or mode=='char2digit', "Wrong mode: %s" % str(mode)
        self.mode = mode

    def processString(self, string):
        if self.mode == 'digit2char':
            return convertDigit2Character(string)
        else:
            return convertCharacter2Digit(string)

    def processFile(self, fileName):
        result = []
        assert os.path.isfile(fileName), "Wrong file path: %s" % str(fileName)
        with codecs.open(fileName,'r','utf-8') as f:
            content=f.readlines()
        if self.mode == 'digit2char':
            for string in content:
                result.append(convertDigit2Character(string))
        else:
            for string in content:
                result.append(convertCharacter2Digit(string))
        return result
        

if __name__ == '__main__':
    DP = DigitProcessor(mode='digit2char')
