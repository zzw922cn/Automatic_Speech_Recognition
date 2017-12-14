#-*- coding:utf-8 -*-
#!/usr/bin/env python3
""" Speech Valley

@author: zzw922cn
@date: 2017-12-02
"""
from speechvalley.feature.core.calcmfcc import calcfeat_delta_delta, calcMFCC
from speechvalley.feature.core.nist2wav import nist2wav
from speechvalley.feature.core.spectrogram import spectrogramPower
