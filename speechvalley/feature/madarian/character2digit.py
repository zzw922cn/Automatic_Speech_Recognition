# encoding: utf-8
# ******************************************************
# Author       : deepxuexi, zzw922cn
# Last modified: 2017-02-01 11:00
# Email        : zzw922cn@gmail.com
# Filename     : madarian_preprocess.py
# Description  : Feature preprocessing for some Madarian dataset
# ******************************************************

import codecs
import re

def _c2n(c_str):
    '''
    Converting character to number
    '''
    if c_str=='':
        return u'0'
    src=u'点零一二三四五六七八九'
    dst=u'.0123456789'
    for i,c in enumerate(src):
        c_str=c_str.replace(c,dst[i])
    return c_str

def _get_gewei(c_str):
    if u'百零' in c_str:
        return _c2n(c_str.split(u'百零')[1])
    elif u'十' in c_str:
        return _c2n(c_str.split(u'十')[1])
    elif u'千零' in c_str:
        return _c2n(c_str.split(u'千零')[1])
    else:
        return '0'

def _get_shiwei(c_str):
    if u'百零' in c_str:
        return u'0'
    elif u'百' in c_str:
        return _c2n(c_str.split(u'百')[1].split(u'十')[0])
    elif u'千零' in c_str and u'十' in c_str:
        return _c2n(c_str.split(u'千零')[1].split(u'十')[0])
    elif u'十' in c_str:
        if c_str.split(u'十')[0]=='':
            return u'1'
        return _c2n(c_str.split(u'十')[0])
    else:
        return u'0'

def _get_baiwei(c_str):
    if u'千零' in c_str:
        return u'0'
    elif u'千' in c_str:
        return _c2n(c_str.split(u'千')[1].split(u'百')[0])
    elif u'百' in c_str:
        return _c2n(c_str.split(u'百')[0])
    else:
        return ''

def _get_qianwei(c_str):
    if u'万零' in c_str:
        return u'0'
    elif u'万' in c_str:
        return _c2n(c_str.split(u'万')[1].split(u'千')[0])
    elif u'千' in c_str:
        return _c2n(c_str.split(u'千')[0])
    else:
        return ''

def _get_complex(c_str):
    gewei=_get_gewei(c_str)
    shiwei=_get_shiwei(c_str)
    baiwei=_get_baiwei(c_str)
    qianwei=_get_qianwei(c_str)
    c_str=qianwei+baiwei+shiwei+gewei
    return c_str

def _convert_to_num(c_str):
    simple=True
    for i in u'十百千万亿':
        if i in c_str:
            simple=False
    if simple:
        return _c2n(c_str)
    else:
        if not u'万' in c_str: #有‘万’的情况
            return _get_complex(c_str)
        else:
            result='' #没‘万’的情况
            for i in c_str.split(u'万'):
                result+=_get_complex(i)
            return result

def _convert_to_nums(c_str):
        if u'亿' in c_str:
            result=''
            for i in c_str.split(u'亿'):
                result+=_convert_to_num(i)
            return result
        elif u'点' in c_str:
            result=[]
            for i in c_str.split(u'点'):
                result.append(_convert_to_num(i))
            return u'.'.join(result)
        else:
            return _convert_to_num(c_str)

def convertCharacter2Digit(string):
    chinese_numbers=re.findall(u'[点零一二三四五六七八九十百千万亿]{2,}', string, re.S)
    sub_str = re.sub(u'[点零一二三四五六七八九十百千万亿]{2,}', '_', string)
    for chinese_number in chinese_numbers:
        digit = _convert_to_nums(chinese_number)
        sub_str = sub_str.replace('_', digit, 1)
    print('原句子:', string)
    print('新句子:', sub_str)
    print('\n')
    return sub_str


if __name__ == '__main__':
    with codecs.open('sample.txt','r','utf-8') as f:
        content=f.readlines()
    for string in content:
        convertCharacter2Digit(string.strip())
