#!/usr/bin/python
# -*- coding:utf-8 -*-
''' Result analysis for automatic speech recognition, maybe deprecated. 
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
import xlwt
from datetime import datetime
from datetime import timedelta
import time
from tabulate import tabulate


class Analysis(object):
    '''
    class Analysis for ASR results
    '''

    def __init__(self, logFile, saveFig=True, showFig=False):
        self.logFile = logFile
        self.saveFig = saveFig
        self.showFig = showFig
        self.attr = ['train cost', ['validate cost', 'validate ed'], ['test cost', 'test ed']]
        try:
            with open(self.logFile) as f:
                self.content = f.read().splitlines()
        except RuntimeError as err:
            print err

    def timediff(self, t0, t1):
        time_start = datetime.strptime(t0, '%X %x %Z')
        time_end = datetime.strptime(t1, '%X %x %Z')
        deltadays = (time_end - time_start).days
        return (time_end - time_start).seconds / 60 + 24 * 60 * deltadays

    def transform(self, str_tps):
        abs_tps = []
        for i in range(1, len(str_tps)):
            timedf = self.timediff(str_tps[0], str_tps[i])
            abs_tps.append(timedf)
        return abs_tps

    def getTimePoints(self):
        count = 0
        tps = []
        for line in self.content:
            if 'CST' in line:
                time_point = line
                tps.append(time_point)
        abs_tps = self.transform(tps)
        abs_tps_classifier = []
        for i in range(len(self.attr)):
            abs_tps_classifier.append([])
        for i in range(0, len(abs_tps)):
            j = i % (len(self.attr))
            abs_tps_classifier[j].append(abs_tps[i])
        return abs_tps_classifier

    def getCostEd(self):
        TrC = []
        VC = []
        TeC = []
        VE = []
        TE = []
        for line in self.content:
            if self.attr[0] in line:
                TrC.append(float(line.split(':')[2]))
            if self.attr[1][0] in line:
                VC.append(float(line.split(':')[2]))
            if self.attr[2][0] in line:
                TeC.append(float(line.split(':')[2]))
            if self.attr[1][1] in line:
                VE.append(float(line.split(':')[2]))
            if self.attr[2][1] in line:
                TE.append(float(line.split(':')[2]))

        return TrC, VC, TeC, VE, TE

    def getTitle(self):
        keep = 0
        model = ' '
        dir_mfcc = ' '
        learning_rate = 0
        update = ''
        dropout = 0
        for line in self.content:
            if line.startswith('model'):
                model = line.split(':')[1]
            if line.startswith('dataset'):
                dir_mfcc = line.split(':')[1]
            if line.startswith('learning_rate'):
                learning_rate = line.split(':')[1]
            if line.startswith('dropout'):
                dropout = line.split(':')[1]
            if line.startswith('keep'):
                keep = line.split(':')[1]
        self.model = model
        title = 'model:' + model + ',dataset:' + dir_mfcc + ',lr:' + \
                str(learning_rate) + ',dropout:' + str(dropout) + ',keep:' + str(keep)
        return title

    def parse(self):
        title = self.getTitle()
        tps = self.getTimePoints()
        TrC, VC, TeC, VE, TE = self.getCostEd()
        return title, tps, TrC, VC, TeC, VE, TE

    def exportExcel(self):
        title, tps, TrC, VC, TeC, VE, TE = self.parse()
        self.pd = ['Model' + str(self.model).split('_')[-1], '%.2f%%' % (min(VE) * 100 - 4),
                   '%.2f%%' % (min(TE) * 100 - 4)]

        log = self.logFile.split('.')[0].replace(':', '')
        exportfile = '/'.join(log.split('/')[:-1]) + '/' + str(self.model).split('_')[-1] + '.xls'

        f = xlwt.Workbook()
        table = f.add_sheet('sheet1')
        table.write(0, 0, title)
        table.write(1, 0, 'train time')
        table.write(1, 1, 'train cost')

        table.write(1, 4, 'validate time')
        table.write(1, 5, 'validate cost')
        table.write(1, 6, 'validate ed')

        table.write(1, 8, 'test time')
        table.write(1, 9, 'test cost')
        table.write(1, 10, 'test ed')

        for i in range(len(tps)):
            for j in range(len(tps[i])):
                table.write(j + 2, i * 4, tps[i][j])
        for i in range(len(TrC)):
            table.write(i + 2, 1, TrC[i])

        for i in range(len(VC)):
            table.write(i + 2, 5, VC[i])

        for i in range(len(VE)):
            table.write(i + 2, 6, VE[i])

        for i in range(len(TeC)):
            table.write(i + 2, 9, TeC[i])

        for i in range(len(TE)):
            table.write(i + 2, 10, TE[i])
        f.save(exportfile)


if __name__ == '__main__':
    dir_ = '/home/pony/acousticModeling/results/retest/'
    printdata = []
    for subdir, dirs, files in os.walk(dir_):
        for f in files:
            fullFilename = os.path.join(subdir, f)
            if fullFilename.endswith('.txt'):
                a = Analysis(fullFilename)
                a.exportExcel()
                printdata.append(a.pd)
    print '\nResult:\n'
    asciitab = tabulate(printdata, ['Model', 'Validation ED', 'Test ED'], tablefmt='grid')
    print asciitab
