#-*- coding:utf-8 -*-
#!/usr/bin/python

''' calculate mfcc feature vectors.
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
from sigprocess import audio2frame
from sigprocess import pre_emphasis
from sigprocess import spectrum_power
from scipy.fftpack import dct 
#首先，为了适配版本3.x，需要调整xrange的使用，因为对于版本2.x只能使用range，需要将xrange替换为range
try:
    xrange(1)
except:
    xrange=range



def calcMFCC_delta_delta(signal,samplerate=16000,win_length=0.025,win_step=0.01,cep_num=13,filters_num=26,NFFT=512,low_freq=0,high_freq=None,pre_emphasis_coeff=0.97,cep_lifter=22,appendEnergy=True):
    '''计算13个MFCC+13个一阶微分系数+13个加速系数,一共39个系数
    '''
    feat=calcMFCC(signal,samplerate,win_length,win_step,cep_num,filters_num,NFFT,low_freq,high_freq,pre_emphasis_coeff,cep_lifter,appendEnergy)   #首先获取13个一般MFCC系数
    feat_delta=delta(feat)
    feat_delta_delta=delta(feat_delta)

    result=numpy.concatenate((feat,feat_delta,feat_delta_delta),axis=1)
    return result

def delta(feat, N=2):
    """Compute delta features from a feature vector sequence.

    :param feat: A numpy array of size (NUMFRAMES by number of features) containing features. Each row holds 1 feature vector.
    :param N: For each frame, calculate delta features based on preceding and following N frames
    :returns: A numpy array of size (NUMFRAMES by number of features) containing delta features. Each row holds 1 delta feature vector.
    """
    NUMFRAMES = len(feat)
    feat = numpy.concatenate(([feat[0] for i in range(N)], feat, [feat[-1] for i in range(N)]))
    denom = sum([2*i*i for i in range(1,N+1)])
    dfeat = []
    for j in range(NUMFRAMES):
        dfeat.append(numpy.sum([n*feat[N+j+n] for n in range(-1*N,N+1)], axis=0)/denom)
    return dfeat
	    
def calcMFCC(signal,samplerate=16000,win_length=0.025,win_step=0.01,cep_num=13,filters_num=26,NFFT=512,low_freq=0,high_freq=None,pre_emphasis_coeff=0.97,cep_lifter=22,appendEnergy=True):
    '''计算13个MFCC系数
    参数含义：
    signal:原始音频信号，一般为.wav格式文件
    samplerate:抽样频率，这里默认为16KHz
    win_length:窗长度，默认即一帧为25ms
    win_step:窗间隔，默认情况下即相邻帧开始时刻之间相隔10ms
    cep_num:倒谱系数的个数，默认为13
    filters_num:滤波器的个数，默认为26
    NFFT:傅立叶变换大小，默认为512
    low_freq:最低频率，默认为0
    high_freq:最高频率
    pre_emphasis_coeff:预加重系数，默认为0.97
    cep_lifter:倒谱的升个数
    appendEnergy:是否加上能量，默认加
    '''
    
    feat,energy=fbank(signal,samplerate,win_length,win_step,filters_num,NFFT,low_freq,high_freq,pre_emphasis_coeff)
    feat=numpy.log(feat)
    feat=dct(feat,type=2,axis=1,norm='ortho')[:,:cep_num]  #进行离散余弦变换,只取前13个系数
    feat=lifter(feat,cep_lifter)
    if appendEnergy:
	feat[:,0]=numpy.log(energy)  #只取2-13个系数，第一个用能量的对数来代替
    return feat

def fbank(signal,samplerate=16000,win_length=0.025,win_step=0.01,filters_num=26,NFFT=512,low_freq=0,high_freq=None,pre_emphasis_coeff=0.97):
    '''计算音频信号的MFCC
    参数说明：
    samplerate:采样频率
    win_length:窗长度
    win_step:窗间隔
    filters_num:梅尔滤波器个数
    NFFT:FFT大小
    low_freq:最低频率
    high_freq:最高频率
    pre_emphasis_coeff:预加重系数
    '''
    
    high_freq=high_freq or samplerate/2  #计算音频样本的最大频率
    signal=pre_emphasis(signal,pre_emphasis_coeff)  #对原始信号进行预加重处理
    frames=audio2frame(signal,win_length*samplerate,win_step*samplerate) #得到帧数组
    spec_power=spectrum_power(frames,NFFT)  #得到每一帧FFT以后的功率谱
    energy=numpy.sum(spec_power,1)  #对每一帧的功率谱进行求和，得到能量
    energy=numpy.where(energy==0,numpy.finfo(float).eps,energy)  #对能量为0的地方调整为eps，这样便于进行对数处理
    fb=get_filter_banks(filters_num,NFFT,samplerate,low_freq,high_freq)  #获得每一个滤波器的频率宽度
    feat=numpy.dot(spec_power,fb.T)  #对滤波器和功率谱进行点乘
    feat=numpy.where(feat==0,numpy.finfo(float).eps,feat)  #同样不能出现0
    return feat,energy
   
def log_fbank(signal,samplerate=16000,win_length=0.025,win_step=0.01,filters_num=26,NFFT=512,low_freq=0,high_freq=None,pre_emphasis_coeff=0.97):
    '''计算对数值
    参数含义：同上
    '''
    feat,energy=fbank(signal,samplerate,win_length,win_step,filters_num,NFFT,low_freq,high_freq,pre_emphasis_coeff)
    return numpy.log(feat)

def ssc(signal,samplerate=16000,win_length=0.025,win_step=0.01,filters_num=26,NFFT=512,low_freq=0,high_freq=None,pre_emphasis_coeff=0.97):
    '''
    待补充
    ''' 
    high_freq=high_freq or samplerate/2
    signal=sigprocess.pre_emphasis(signal,pre_emphasis_coeff)
    frames=sigprocess.audio2frame(signal,win_length*samplerate,win_step*samplerate)
    spec_power=sigprocess.spectrum_power(frames,NFFT) 
    spec_power=numpy.where(spec_power==0,numpy.finfo(float).eps,spec_power) #能量谱
    fb=get_filter_banks(filters_num,NFFT,samplerate,low_freq,high_freq) 
    feat=numpy.dot(spec_power,fb.T)  #计算能量
    R=numpy.tile(numpy.linspace(1,samplerate/2,numpy.size(spec_power,1)),(numpy.size(spec_power,0),1))
    return numpy.dot(spec_power*R,fb.T)/feat

def hz2mel(hz):
    '''把频率hz转化为梅尔频率
    参数说明：
    hz:频率
    '''
    return 2595*numpy.log10(1+hz/700.0)

def mel2hz(mel):
    '''把梅尔频率转化为hz
    参数说明：
    mel:梅尔频率
    '''
    return 700*(10**(mel/2595.0)-1)

def get_filter_banks(filters_num=20,NFFT=512,samplerate=16000,low_freq=0,high_freq=None):
    '''计算梅尔三角间距滤波器，该滤波器在第一个频率和第三个频率处为0，在第二个频率处为1
    参数说明：
    filers_num:滤波器个数
    NFFT:FFT大小
    samplerate:采样频率
    low_freq:最低频率
    high_freq:最高频率
    '''
    #首先，将频率hz转化为梅尔频率，因为人耳分辨声音的大小与频率并非线性正比，所以化为梅尔频率再线性分隔
    low_mel=hz2mel(low_freq)
    high_mel=hz2mel(high_freq)
    #需要在low_mel和high_mel之间等间距插入filters_num个点，一共filters_num+2个点
    mel_points=numpy.linspace(low_mel,high_mel,filters_num+2)
    #再将梅尔频率转化为hz频率，并且找到对应的hz位置
    hz_points=mel2hz(mel_points)
    #我们现在需要知道这些hz_points对应到fft中的位置
    bin=numpy.floor((NFFT+1)*hz_points/samplerate)
    #接下来建立滤波器的表达式了，每个滤波器在第一个点处和第三个点处均为0，中间为三角形形状
    fbank=numpy.zeros([filters_num,NFFT/2+1])
    for j in xrange(0,filters_num):
	for i in xrange(int(bin[j]),int(bin[j+1])):
	    fbank[j,i]=(i-bin[j])/(bin[j+1]-bin[j])
	for i in xrange(int(bin[j+1]),int(bin[j+2])):
	    fbank[j,i]=(bin[j+2]-i)/(bin[j+2]-bin[j+1])
    return fbank

def lifter(cepstra,L=22):
    '''升倒谱函数
    参数说明：
    cepstra:MFCC系数
    L：升系数，默认为22
    '''
    if L>0:
	nframes,ncoeff=numpy.shape(cepstra)
	n=numpy.arange(ncoeff)
	lift=1+(L/2)*numpy.sin(numpy.pi*n/L)
	return lift*cepstra
    else:
	return cepstra
