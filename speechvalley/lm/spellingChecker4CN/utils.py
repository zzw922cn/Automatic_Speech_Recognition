# encoding: utf-8
# ******************************************************
# Author       : zzw922cn
# Last modified: 2017-12-20 11:00
# Email        : zzw922cn@gmail.com
# Filename     : utils.py
# Description  : Spelling Corrector for Chinese
# ******************************************************

import string
import re
def filter_punctuation(input_str, remove_duplicate_space=True):
    """
    all common punctuations in both Chinese and English, if any marker is 
    not included, welcome to pull issues in github repo.
    """
    '''
    punctuation=string.punctuation + string.ascii_letters + \
                '！？｡＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀' + \
                '｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞' + \
                '〟〰〾〿–—‘’‛“”„‟…‧﹏.·。《》'
    regex = re.compile('[%s]' % re.escape(punctuation))
    '''

    regex = re.compile(u'[^\u4E00-\u9FA5]')#非中文
    if remove_duplicate_space:
        result = re.sub(' +', ' ', regex.sub(' ', input_str))
    else:
        result = regex.sub(' ', input_str)
    result = re.sub("\d+", " ", result)
    result = strQ2B(result)
    return result

def strQ2B(ustring):
    """
    Converting full-width characters to half-width characters
    """
    rstring = ""
    for uchar in ustring:
        inside_code = ord(uchar)
        if inside_code == 12288: 
            inside_code = 32
        elif (inside_code >= 65281 and inside_code <= 65374): 
            inside_code -= 65248
        rstring += chr(inside_code)
    return rstring

def digits2Chinese(string, mode='int'):
    if mode == 'integer':
    

def reviseString(string):
    re_int = re.compile('\d+')
    re_float = re.compile('\d+\.\d+')



if __name__ == '__main__':
    a = 'abcd我是，,,,...上升！!!!~[][][]·「·」「{}345'
    print(filter_punctuation(a))
