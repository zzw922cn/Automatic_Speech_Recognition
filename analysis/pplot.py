#-*- coding:utf-8 -*-
#!/usr/bin/python

''' This file is designed to plot the cost curve, maybe deprecated.
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


import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

class Analysis(object):
    '''
    class Analysis for ASR results
    '''
    def __init__(self,logFile,saveFig=True,showFig=False):
        self.logFile = logFile
        self.saveFig = saveFig
        self.showFig = showFig
    def getContent(self):
        try:
            with open(self.logFile) as f:
                content = f.read().splitlines()
        except RuntimeError as err:
            print err
            return content
    def parse(self):
        indexCostList = []
        index1 = 0
        indexValidateList = []
        index2 = 0
        costList = []
        validateCostList = []
        content = self.getContent()
        keep = 0
        model = ' '
        dir_mfcc = ' '
        learning_rate = 0
        update = ''
        for line in content:
            if line.startswith('model'):
                model = line.split(':')[1]
            if line.startswith('penalty'):
                penalty = line.split(':')[1]
            if line.startswith('input_dim'):
                input_dim = line.split(':')[1]
            if line.startswith('n_hid'):
                n_hid = line.split(':')[1]
            if line.startswith('dataset'):
                dir_mfcc = line.split(':')[1]
            if line.startswith('learning_rate'):
                learning_rate = line.split(':')[1]
            if line.startswith('update'):
                update = line.split(' ')[2]
            if line.startswith('keep'):
                keep = line.split(':')[1]
            if line.startswith('Epoch'):
                if 'validate cost' in line:
                    index2 = index2 + 1
                    cost = line.split(':')[2]
                    indexValidateList.append(index2)
                    validateCostList.append(float(cost))
                elif 'train cost' in line:
                    index1 = index1+1
                    cost = line.split(':')[2]
                    indexCostList.append(index1)
                    costList.append(float(cost))
    title = 'model:'+model+',dataset:'+dir_mfcc+',lr:'+ \
        str(learning_rate)+'\nupdate:'+update
    return title,indexCostList,indexValidateList,costList,validateCostList

    def plot(self):
        title,indexCostList,indexValidateList,costList,validateCostList = self.parse()
        p1 = plt.plot(indexCostList,costList,marker='o',color='b',label='train cost')
        p2 = plt.plot(indexValidateList,validateCostList,marker='o',color='r',label='validate cost')
        plt.xlabel('Epoch')
        plt.ylabel('Cost')
        plt.legend()
        plt.grid()
        plt.title(title)
        if self.saveFig:
            plt.savefig(self.logFile+'.png',dpi=100)
            #plt.savefig(self.logFile+'.eps',dpi=100)
        if self.showFig:
            plt.show()

if __name__ == '__main__':
    dir_ = '/home/pony/acousticModeling/results/retest/'
    for subdir, dirs, files in os.walk(dir_):
        for f in files:
            fullFilename = os.path.join(subdir, f)
            if fullFilename.endswith('.txt'):
                a = Analysis(fullFilename)
                a.plot()
                plt.clf()
