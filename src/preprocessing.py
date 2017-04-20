import os

import numpy as np
import scipy.io.wavfile as wav

from feature.calcmfcc import calcMFCC_delta_delta

PHN_LOOKUP_TABLE = ['aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ax-h', 'axr', 'ay', 'b', 'bcl', 'ch', 'd', 'dcl', 'dh', 'dx',
                    'eh', 'el', 'em', 'en', 'eng', 'epi', 'er', 'ey', 'f', 'g', 'gcl', 'h#', 'hh', 'hv', 'ih', 'ix',
                    'iy', 'jh', 'k', 'kcl', 'l', 'm', 'n', 'ng', 'nx', 'ow', 'oy', 'p', 'pau', 'pcl', 'q', 'r', 's',
                    'sh', 't', 'tcl', 'th', 'uh', 'uw', 'ux', 'v', 'w', 'y', 'z', 'zh']


def wav_to_mfcc(wav_file_path):
    rate, sig = wav.read(wav_file_path)
    mfcc = calcMFCC_delta_delta(sig, rate, win_length=0.020, win_step=0.010)
    mfcc = np.transpose(mfcc)
    return mfcc


def create_label(phn_file_path):
    phenome = []
    with open(phn_file_path, 'r') as f:
        for line in f.read().splitlines():
            s = line.split(' ')[2]
            p_index = PHN_LOOKUP_TABLE.index(s)
            phenome.append(p_index)
    return np.array(phenome)


def transform_raw_data(raw_data_dir, dest_mfcc_dir, dest_label_dir):
    i = 0
    for subdir, dirs, files in os.walk(raw_data_dir):
        for file in files:
            if file.endswith('.wav'):
                wav_file_path = os.path.join(subdir, file)
                speech_file_base = os.path.splitext(wav_file_path)[0]
                phn_file_path = speech_file_base + '.phn'
                mfcc_file_path = dest_mfcc_dir + '-'.join(speech_file_base.split('/')[-2:]) + '.npy'
                label_file_path = dest_label_dir + '-'.join(speech_file_base.split('/')[-2:]) + '.npy'

                print('[{}] processing: {}'.format(i, wav_file_path))
                mfcc = wav_to_mfcc(wav_file_path)
                np.save(mfcc_file_path, mfcc)

                phenome = create_label(phn_file_path)
                np.save(label_file_path, np.array(phenome))

                i += 1
