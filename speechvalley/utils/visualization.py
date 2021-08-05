# encoding: utf-8
# ******************************************************
# Author       : zzw922cn
# Last modified: 2017-12-23 11:00
# Email        : zzw922cn@gmail.com
# Filename     : visualization.py
# Description  : Some common functions for visualization
# ******************************************************

import matplotlib.pyplot as plt
import numpy as np
import wave
import sys

def plotWaveform(audio):
    """
    Plotting the waveform of a wav file
    """
    spf = wave.open(audio,'r')
    signal = spf.readframes(-1)
    signal = np.fromstring(signal, 'Int16')
    fs = spf.getframerate()
    if spf.getnchannels() == 2:
        print('Just mono files')
        sys.exit(0)
    Time=np.linspace(0, len(signal)/fs, num=len(signal))
    plt.figure(1)
    plt.title('Waveform of an audio')
    plt.plot(Time,signal)
    plt.show()
