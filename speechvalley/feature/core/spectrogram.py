# encoding: utf-8
# ******************************************************
# Author       : zzw922cn
# Last modified: 2017-12-09 11:00
# Email        : zzw922cn@gmail.com
# Filename     : spectrogram.py
# Description  : Calculating spectrogram for Automatic Speech Recognition
# ******************************************************

import numpy as np
import scipy.io.wavfile as wav
import librosa
from sklearn import preprocessing

def spectrogramPower(audio, window_size=0.02, window_stride=0.01):
    """ short time fourier transform

    Details:
        audio - This is the input time-domain signal you wish to find the spectrogram of. It can't get much simpler than that. In your case, the 
                signal you want to find the spectrogram of is defined in the following code:

        win_length - If you recall, we decompose the image into chunks, and each chunk has a specified width.  window defines the width of each 
                 chunkin terms of samples. As this is a discrete-time signal, you know that this signal was sampled with a particular sampling 
                 frequency and sampling period. You can determine how large the window is in terms of samples by:

                 window_samples = window_time/Ts
        hop_length - the same as stride in convolution network, overlapping width

    """
    samplingRate, samples = wav.read(audio)
    win_length = int(window_size * samplingRate)
    hop_length = int(window_stride * samplingRate)
    n_fft = win_length
    D = librosa.core.stft(samples, n_fft=n_fft,hop_length=hop_length,
                      win_length=win_length)
    mag = np.abs(D)
    log_mag = np.log1p(mag)
    # normalization
    log_mag = preprocessing.scale(log_mag)
    # size: frequency_bins*time_len
    return log_mag
    
    
if __name__ == '__main__':
    print(np.shape(spectrogramPower('test.wav')))
