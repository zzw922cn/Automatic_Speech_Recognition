# -*- coding:utf-8 -*-
''' data visualization for automatic speech recognition
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

date:2017-3-14
'''

import os
import time
import numpy as np
import matplotlib.pyplot as plt

def readlogs(rootdir):
    ''' function for reading asr logs to visualize'''
    trainERs = []
    testERs = []
    fullFilenames = []
    for subdir, dirs, files in os.walk(rootdir):
        for f in files:
            fullFilename = os.path.join(subdir, f)
            fullFilenames.append(fullFilename)
    fullFilenames.sort(key=lambda x: os.path.getctime(x))
    print fullFilenames
    epoch = 0
    if True:
        for fullFilename in fullFilenames:
            if fullFilename.endswith('.txt'):
                with open(fullFilename, 'r') as train_file:
                    content = train_file.readlines()
                for line in content:
                    if 'train error rate' in line:
                        trainER = line.split(':')[2]
                        trainERs.append(float(trainER))
                        epoch += 1
            elif fullFilename.endswith('TEST'):
                with open(fullFilename, 'r') as test_file:
                    content = test_file.readlines()
                for line in content:
                    if 'test error rate' in line:
                        testER = line.split(':')[1]
                        testERs.append(float(testER))

    return trainERs, testERs

def visualize(trainERs, testERs):
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    ax1.plot(trainERs, label='train phoneme error rate')
    ax2.plot(testERs, label='test phoneme error rate')
    ax1.legend()
    ax2.legend()
    ax1.grid()
    ax2.grid()
    plt.suptitle('dynamic bidirectional LSTM for Automatic Speech Recognition')
    plt.show()
if __name__ == '__main__':
    rootdir = '/home/pony/github/data/ASR/log/'
    train, test = readlogs(rootdir)
    visualize(train, test)
