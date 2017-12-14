# encoding: utf-8
# ******************************************************
# Author       : zzw922cn
# Last modified: 2017-12-09 11:00
# Email        : zzw922cn@gmail.com
# Filename     : ngram.py
# Description  : ngram model 
# ******************************************************

import numpy as np
import os
import operator
import pickle

def save_obj(name, obj):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

class NGram:
    def __init__(self, rootdir):
        self.rootdir = rootdir

    def get_corpus(self):
        corpus = []
        word_count = {}
        biword_count = {}
        bigram = {}
        bigram['SOS'] = {}
        trigram = {}
        for subdir, dirs, files in os.walk(self.rootdir):
            for f in files:
                fullFilename = os.path.join(subdir, f)
                filenameNoSuffix =  os.path.splitext(fullFilename)[0]
            if f.endswith('.label'):
                with open(fullFilename, 'r') as f:
                    line = f.readline()
                    corpus.append(line)
                    line = line.strip().split(' ')
                    len_sent = range(len(line))
                    for idx in len_sent:
                        word = line[idx]
                        word_count = inc_dict(word_count, word)
                        if not bigram.has_key(word):
                            bigram[word] = {}
                        if idx == 0:
                            bigram['SOS'] = inc_dict(bigram['SOS'], word)
                        elif idx != len(line)-1:
                            bigram[word] = inc_dict(bigram[word], line[idx+1])
                        else:
                            bigram[word] = inc_dict(bigram[word], 'EOS')
                        if idx == 0:
                            tri_key = 'SOS ' + word
                        else:
                            tri_key = line[idx-1]+' '+word
                        if not trigram.has_key(tri_key):
                            trigram[tri_key] = {}
                        if idx == len(line)-1:
                            trigram[tri_key] = inc_dict(trigram[tri_key], 'EOS')
                        else:
                            trigram[tri_key] = inc_dict(trigram[tri_key], line[idx+1])

        return corpus, word_count, bigram, trigram

def inc_dict(dic, key):
    if not dic.has_key(key):
        dic[key] = 0
        dic[key] += 1
    return dic


if __name__ == '__main__':
    ngram = NGram('/media/pony/Seagate Expansion Drive/学习/语音识别/ASR数据库/LibriSpeech/')
    corpus, word_count, bigram, trigram = ngram.get_corpus()
    savedir = '/home/pony/github/data/libri/ngram/'
    save_obj(savedir+'corpus', corpus)
    save_obj(savedir+'word_count', word_count)
    save_obj(savedir+'bigram', bigram)
    save_obj(savedir+'trigram', trigram)
    #sorted_word_count = sorted(word_count.items(), key=operator.itemgetter(1), reverse=True)
