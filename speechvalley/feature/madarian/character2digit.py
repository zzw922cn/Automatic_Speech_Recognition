#coding=utf-8
import codecs,re
with codecs.open('sample.txt','r','utf-8') as f:
    content=f.read()
chinese_numbers=re.findall(u'[点零一二三四五六七八九十百千万亿]{2,}',content,re.S) #提取所有中文表示的数字

def c2n(c_str):
    if c_str=='':
        return u'0'
    src=u'点零一二三四五六七八九'
    dst=u'.0123456789'
    for i,c in enumerate(src):
        c_str=c_str.replace(c,dst[i])
    return c_str

def get_gewei(c_str):
    if u'百零' in c_str:
        return c2n(c_str.split(u'百零')[1])
    elif u'十' in c_str:
        return c2n(c_str.split(u'十')[1])
    elif u'千零' in c_str:
        return c2n(c_str.split(u'千零')[1])
    else:
        return '0'

def get_shiwei(c_str):
    if u'百零' in c_str:
        return u'0'
    elif u'百' in c_str:
        return c2n(c_str.split(u'百')[1].split(u'十')[0])
    elif u'千零' in c_str and u'十' in c_str:
        return c2n(c_str.split(u'千零')[1].split(u'十')[0])
    elif u'十' in c_str:
        if c_str.split(u'十')[0]=='':
            return u'1'
        return c2n(c_str.split(u'十')[0])
    else:
        return u'0'

def get_baiwei(c_str):
    if u'千零' in c_str:
        return u'0'
    elif u'千' in c_str:
        return c2n(c_str.split(u'千')[1].split(u'百')[0])
    elif u'百' in c_str:
        return c2n(c_str.split(u'百')[0])
    else:
        return ''

def get_qianwei(c_str):
    if u'万零' in c_str:
        return u'0'
    elif u'万' in c_str:
        return c2n(c_str.split(u'万')[1].split(u'千')[0])
    elif u'千' in c_str:
        return c2n(c_str.split(u'千')[0])
    else:
        return ''

def get_complex(c_str):
    gewei=get_gewei(c_str)
    shiwei=get_shiwei(c_str)
    baiwei=get_baiwei(c_str)
    qianwei=get_qianwei(c_str)
    c_str=qianwei+baiwei+shiwei+gewei
    # print(qianwei,baiwei,shiwei,gewei)
    return c_str

def convert_to_num(c_str):
    simple=True
    for i in u'十百千万亿':
        if i in c_str:
            simple=False
    if simple:
        return c2n(c_str)
    else:
        if not u'万' in c_str: #有‘万’的情况
            return get_complex(c_str)
        else:
            result='' #没‘万’的情况
            for i in c_str.split(u'万'):
                result+=get_complex(i)
            return result

def convert_to_nums(c_str):
        if u'亿' in c_str:
            result=''
            for i in c_str.split(u'亿'):
                result+=convert_to_num(i)
            return result
        elif u'点' in c_str:
            result=[]
            for i in c_str.split(u'点'):
                result.append(convert_to_num(i))
            return u'.'.join(result)
        else:
            return convert_to_num(c_str)

for chinese_number in chinese_numbers:
    print('原句子:'+chinese_number,'\n新句子:', convert_to_nums(chinese_number))
    print('\n')
