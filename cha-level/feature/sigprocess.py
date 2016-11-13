#-*- coding:utf-8 -*-
#!/usr/bin/python

''' 
author:

      iiiiiiiiiiii            iiiiiiiiiiii         !!!!!!!             !!!!!!    
      #        ###            #        ###           ###        I#        #:     
      #      ###              #      I##;             ##;       ##       ##      
            ###                     ###               !##      ####      #       
           ###                     ###                 ###    ## ###    #'       
         !##;                    `##%                   ##;  ##   ###  ##        
        ###                     ###                     $## `#     ##  #         
       ###        #            ###        #              ####      ####;         
     `###        -#           ###        `#               ###       ###          
     ##############          ##############               `#         #     
     
date:2016-11-09
'''


import numpy
import math

def audio2frame(signal,frame_length,frame_step,winfunc=lambda x:numpy.ones((x,))):
    '''将音频信号转化为帧，以采样点为单位。
	参数含义：
	signal:原始音频型号
	frame_length:每一帧的长度(这里指采样点的长度，即采样频率乘以时间间隔)
	frame_step:相邻帧的间隔（同上定义）
	winfunc:lambda函数，用于生成一个向量
    '''
    signal_length=len(signal) #信号总长度
    frame_length=int(round(frame_length)) #以帧帧时间长度
    frame_step=int(round(frame_step)) #相邻帧之间的步长
    if signal_length<=frame_length: #若信号长度小于一个帧的长度，则帧数定义为1
        frames_num=1
    else: #否则，计算帧的总长度
        frames_num=1+int(math.ceil((1.0*signal_length-frame_length)/frame_step))
    pad_length=int((frames_num-1)*frame_step+frame_length) #所有帧加起来总的铺平后的长度
    zeros=numpy.zeros((pad_length-signal_length,)) #不够的长度使用0填补，类似于FFT中的扩充数组操作
    pad_signal=numpy.concatenate((signal,zeros)) #填补后的信号记为pad_signal
    indices=numpy.tile(numpy.arange(0,frame_length),(frames_num,1))+numpy.tile(numpy.arange(0,frames_num*frame_step,frame_step),(frame_length,1)).T  #相当于对所有帧的时间点进行抽取，得到frames_num*frame_length长度的矩阵
    indices=numpy.array(indices,dtype=numpy.int32) #将indices转化为矩阵
    frames=pad_signal[indices] #得到帧信号
    win=numpy.tile(winfunc(frame_length),(frames_num,1))  #window窗函数，这里默认取1
    return frames*win   #返回帧信号矩阵

def deframesignal(frames,signal_length,frame_length,frame_step,winfunc=lambda x:numpy.ones((x,))):
    '''定义函数对原信号的每一帧进行变换，应该是为了消除关联性
    参数定义：
    frames:audio2frame函数返回的帧矩阵
    signal_length:信号长度
    frame_length:帧长度
    frame_step:帧间隔
    winfunc:对每一帧加window函数进行分析，默认此处不加window
    '''
    #对参数进行取整操作
    signal_length=round(signal_length) #信号的长度
    frame_length=round(frame_length) #帧的长度
    frames_num=numpy.shape(frames)[0] #帧的总数
    assert numpy.shape(frames)[1]==frame_length,'"frames"矩阵大小不正确，它的列数应该等于一帧长度'  #判断frames维度，处理异常
    indices=numpy.tile(numpy.arange(0,frame_length),(frames_num,1))+numpy.tile(numpy.arange(0,frames_num*frame_step,frame_step),(frame_length,1)).T  #相当于对所有帧的时间点进行抽取，得到frames_num*frame_length长度的矩阵
    indices=numpy.array(indices,dtype=numpy.int32)
    pad_length=(frames_num-1)*frame_step+frame_length #铺平后的所有信号
    if signal_length<=0:
        signal_length=pad_length
    recalc_signal=numpy.zeros((pad_length,)) #调整后的信号
    window_correction=numpy.zeros((pad_length,1)) #窗关联
    win=winfunc(frame_length)
    for i in range(0,frames_num):
        window_correction[indices[i,:]]=window_correction[indices[i,:]]+win+1e-15 #表示信号的重叠程度
        recalc_signal[indices[i,:]]=recalc_signal[indices[i,:]]+frames[i,:] #原信号加上重叠程度构成调整后的信号
    recalc_signal=recalc_signal/window_correction #新的调整后的信号等于调整信号处以每处的重叠程度
    return recalc_signal[0:signal_length] #返回该新的调整信号

def spectrum_magnitude(frames,NFFT):
    '''计算每一帧经过FFY变换以后的频谱的幅度，若frames的大小为N*L,则返回矩阵的大小为N*NFFT
    参数说明：
    frames:即audio2frame函数中的返回值矩阵，帧矩阵
    NFFT:FFT变换的数组大小,如果帧长度小于NFFT，则帧的其余部分用0填充铺满
    '''
    complex_spectrum=numpy.fft.rfft(frames,NFFT) #对frames进行FFT变换
    return numpy.absolute(complex_spectrum)  #返回频谱的幅度值

def spectrum_power(frames,NFFT):
    '''计算每一帧傅立叶变换以后的功率谱
    参数说明：
    frames:audio2frame函数计算出来的帧矩阵
    NFFT:FFT的大小
    '''
    return 1.0/NFFT * numpy.square(spectrum_magnitude(frames,NFFT)) #功率谱等于每一点的幅度平方/NFFT

def log_spectrum_power(frames,NFFT,norm=1):
    '''计算每一帧的功率谱的对数形式
    参数说明：
    frames:帧矩阵，即audio2frame返回的矩阵
    NFFT：FFT变换的大小
    norm:范数，即归一化系数
    '''
    spec_power=spectrum_power(frames,NFFT)
    spec_power[spec_power<1e-30]=1e-30 #为了防止出现功率谱等于0，因为0无法取对数
    log_spec_power=10*numpy.log10(spec_power)
    if norm:
        return log_spec_power-numpy.max(log_spec_power)
    else:
        return log_spec_power

def pre_emphasis(signal,coefficient=0.95):
    '''对信号进行预加重
    参数含义：
    signal:原始信号
    coefficient:加重系数，默认为0.95
    '''
    return numpy.append(signal[0],signal[1:]-coefficient*signal[:-1])



