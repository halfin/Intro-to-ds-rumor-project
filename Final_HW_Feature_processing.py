# coding:utf-8
import numpy as np
import json
import os
import jieba
import jieba.analyse
import re
from snownlp import SnowNLP

##parameter
data_place = r"C:\Users\lenovo\Desktop\original-microblog"

##hyper-parameter
top_word = 20##分词超参数
allowpos = ('ns', 'n', 'vn', 'v','a','e','d','o','r')##关键词属性设置
##feature
##content feature
corpus = []##语料库
excalmatory_mark = []##感叹号个数
question_mark = []##问号个数
postive_words = []##褒义词个数
negative_words = []##贬义词个数
positive_rate = []##这条微博的积极性
##weibo feature
has_url = []##微博当中是否含有超链接
##user feature
description = []##原始微博发布者是否有描述
gender = []##原始微博发布者性别
followers = []##原始微博发布者被关注量
friends = []##原始微博发布者朋友数
vertified = []##用户是否认证
##label
category = []##类别

##共3300个数据，1451个谣言
def read_data(path):
    global corpus
    global gender
    global followers
    global friends
    global vertified
    global description
    global has_url
    global category
    path_list = os.listdir(path)
    for filename in path_list:
        if os.path.splitext(filename)[1] == '.json':
            with open(path + "\\" + filename, encoding='utf-8') as fp:
                data = json.load(fp)
                if(data['user'] == 'empty'):
                    continue
                corpus.append(re.sub(r'[^\u4E00-\u9FA5?？!！]',"",data['text']))
                description.append(data['user']['description'])
                gender.append(data['user']['gender'])
                followers.append(data['user']['followers'])
                friends.append(data['user']['friends'])
                has_url.append(data['has_url'])
                vertified.append(data['user']['verified'])
                if(int(re.match("\d+",filename).group()) <= 2600):
                    category.append(1)
                if(int(re.match("\d+",filename).group()) > 2600):
                    category.append(0)

    has_url = np.array(has_url).astype(int)
    vertified = np.array(vertified).astype(int)
    description = np.array(description).astype(int)
    gender = np.array(list(map(trans_gender, gender)))
    followers = np.array(followers)
    friends = np.array(friends)
    category = np.array(category)

def trans_gender(gender):
    if(gender == "m"):
        return 1
    else:
        return 0
##统计感叹号与问号的个数
def count_punction():
    global excalmatory_mark
    global question_mark
    excalmatory_mark = np.array([i.count('！') + i.count('!') for i in corpus])
    question_mark = np.array([i.count('？') + i.count('?') for i in corpus])

## transform sentence into several words
# def cut_word(text):
#     return " ".join(list(jieba.cut(text)))
# def corpus2new_corpus():
#     global new_corpus
#     global corpus
#     for sent in corpus:
#         sent = re.sub(r'[^\u4E00-\u9FA5]', "", sent)
#         new_corpus.append(cut_word(sent))

##acquire text features
##处理原始微博的褒贬属性(统计褒贬词的个数[根据词的情感态度]，分析一条微博的积极性，算法为——
##将一条微博的前top_word个关键词找出，其排序的方式为TF-IDF，TF-IDF对应的总语料库为jieba分词总语料库
##将这些词(情感系数-0.5)关于其重要性加权平均得到这条微博的积极性)
def handle_corpus():
    global positive_rate
    global postive_words
    global negative_words
    global corpus
    for sent in corpus:
        postive_num = 0
        negative_num = 0
        rate = 0
        for x, w in jieba.analyse.textrank(sent, topK=top_word, withWeight=True,allowPOS=allowpos):
            rate += (SnowNLP(x).sentiments - 0.5) * w
            if (SnowNLP(x).sentiments < 0.4):
                negative_num += 1
            if (SnowNLP(x).sentiments > 0.6):
                postive_num += 1
        ##加上最后一个0.000000001是因为有的微博没有关键词，所以为了防止分母为0加上一个微小数
        rate = rate/(len(jieba.analyse.textrank(sent, topK=top_word, withWeight=True, allowPOS=allowpos))+0.000000001)
        postive_words.append(postive_num)
        negative_words.append(negative_num)
        positive_rate.append(rate)
    postive_words = np.array(postive_words)
    negative_words = np.array(negative_words)
    positive_rate = np.array(positive_rate)


##read json document and acquire corpus
read_data(data_place)
## Given a corpus:
count_punction()
handle_corpus()