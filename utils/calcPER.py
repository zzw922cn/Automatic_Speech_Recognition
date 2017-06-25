#coding=utf-8
'''Calculating Phoneme Error Rate(PER) based on python leven, rather
than build a new session based on tf.edit_distance

author:
zzw922cn,brianlan

2017-6-24
'''

from collections import namedtuple
import leven
import numpy as np

SparseTensor = namedtuple('SparseTensor', 'indices vals shape')

PHN_MAPPING = {'iy': 'iy', 'ix': 'ix', 'ih': 'ix', 'eh': 'eh', 'ae': 'ae', 'ax': 'ax', 'ah': 'ax',
               'ax-h': 'ax', 'uw': 'uw', 'ux': 'uw', 'uh': 'uh', 'ao': 'ao', 'aa': 'ao', 'ey': 'ey',
               'ay': 'ay', 'oy': 'oy', 'aw': 'aw', 'ow': 'ow', 'er': 'er', 'axr': 'er', 'l': 'l', 'el': 'l',
               'r': 'r', 'w': 'w', 'y': 'y', 'm': 'm', 'em': 'm', 'n': 'n', 'en': 'n', 'nx': 'n', 'ng': 'ng',
               'eng': 'ng', 'v': 'v', 'f': 'f', 'dh': 'dh', 'th': 'th', 'z': 'z', 's': 's', 'zh': 'zh',
               'sh': 'zh', 'jh': 'jh', 'ch': 'ch', 'b': 'b', 'p': 'p', 'd': 'd', 'dx': 'dx', 't': 't',
               'g': 'g', 'k': 'k', 'hh': 'hh', 'hv': 'hh', 'bcl': 'h#', 'pcl': 'h#', 'dcl': 'h#', 'tcl': 'h#',
               'gcl': 'h#', 'kcl': 'h#', 'q': 'h#', 'epi': 'h#', 'pau': 'h#', 'h#': 'h#'}

IDX_MAPPING = {0: 3, 1: 1, 2: 5, 3: 3, 4: 4, 5: 5, 6: 5, 7: 22, 8: 8, 9: 9, 10: 27, 11: 11, 12: 12, 13: 27,
               14: 14, 15: 15, 16: 16, 17: 36, 18: 37, 19: 38, 20: 39, 21: 27, 22: 22, 23: 23, 24: 24, 25: 25,
               26: 27, 27: 27, 28: 28, 29: 28, 30: 31, 31: 31, 32: 32, 33: 33, 34: 34, 35: 27, 36: 36, 37: 37,
               38: 38, 39: 39, 40: 38, 41: 41, 42: 42, 43: 43, 44: 27, 45: 27, 46: 27, 47: 47, 48: 48, 49: 60,
               50: 50, 51: 27, 52: 52, 53: 53, 54: 54, 55: 54, 56: 56, 57: 57, 58: 58, 59: 59, 60: 60}


def calc_PER(pred, ground_truth, normalize=True, merge_phn=True):
    """Calculates the Phoneme Error Rate based on python package leven, which produce the same results as 
    tf.edit_distance and tf.reduce_mean based calculation
    
    :param pred: tuple with 3 numpy-typed element representing sparse tensor
    :param ground_truth: tuple with 3 numpy-typed element representing sparse tensor
    :param normalize: if True, the distance between sequence will be divided by the length of the ground_truth length
    :param merge_phn: if True, 61 phonemes will be merged into 39 phonemes, then do the distance calculation
    :return: the PER
    """

    pred_seq_list = seq_to_single_char_strings(sparse_tensor_to_seq_list(pred, merge_phn=merge_phn))
    truth_seq_list = seq_to_single_char_strings(sparse_tensor_to_seq_list(ground_truth, merge_phn=merge_phn))

    assert len(truth_seq_list) == len(pred_seq_list)

    distances = []
    for i in range(len(truth_seq_list)):
        dist_i = leven.levenshtein(pred_seq_list[i], truth_seq_list[i])
        if normalize:
            dist_i /= float(len(truth_seq_list[i]))
        distances.append(dist_i)

    return np.mean(distances)


def seq_to_single_char_strings(seq):
    strings = []
    for s in seq:
        strings.append(''.join([chr(65 + p) for p in s]))

    return strings


def sparse_tensor_to_seq_list(sparse_seq, merge_phn=True):
    phonemes_list = []
    it = 0
    num_samples = np.max(sparse_seq.indices, axis=0)[0] + 1
    for n in range(num_samples):
        cur_sample_indices = sparse_seq.indices[sparse_seq.indices[:, 0] == n, 1]

        if len(cur_sample_indices) == 0:
            seq_length = 0
        else:
            seq_length = np.max(cur_sample_indices) + 1

        seq = sparse_seq.vals[it:it+seq_length]
        _seq = [IDX_MAPPING[p] for p in seq] if merge_phn else seq
        phonemes_list.append(_seq)
        it += seq_length

    return phonemes_list

