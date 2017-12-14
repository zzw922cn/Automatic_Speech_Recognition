# encoding: utf-8
# ******************************************************
# Author       : zzw922cn
# Last modified: 2017-12-09 11:00
# Email        : zzw922cn@gmail.com
# Filename     : generate.py
# Description  : Generation for characters by n-gram model
# ******************************************************

import numpy as np
import pickle

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def frequence(gram, type=2):
    if type == 2:
        for key, value in gram.items():
            total = 0.0
            for subkey, subvalue in value.items():
                total += subvalue
            for subkey, subvalue in value.items():
                gram[key][subkey] = subvalue/total
    else:
        raise NotImplementedError('%s-gram is being developed'%type)
    return gram

def generate_sentence(corpus_dir, seed='what are', length=10):
    bigram = load_obj(corpus_dir+'bigram')
    freq_bigram = frequence(bigram)
    sent = ''
    if not ' ' in seed:
        sent += seed
        prev = seed
        for i in range(length):
            probs = []
            for _, value in freq_bigram[prev].items():
                probs.append(value)
            sample = np.random.choice(range(len(freq_bigram[prev])),p=probs)
            prev = freq_bigram[prev].keys()[sample]
            sent += ' '+prev
    print sent

generate_sentence('/home/pony/github/data/libri/ngram/', seed='love', length=10)
