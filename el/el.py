#coding=utf-8
import pymongo
import json
import os
import re
from itertools import groupby
from random import choice
import time
import jieba
import numpy as np
import requests
from gensim.models import Word2Vec
from nlp_zero import Trie, DAG  # pip install nlp_zero
from tqdm import tqdm



char_size = 128
num_features = 2

word2vec = Word2Vec.load('word2vec_baike/word2vec_baike')
id2word = {i + 1: j for i, j in enumerate(word2vec.wv.index2word)}
word2id = {j: i for i, j in id2word.items()}
word2vec = word2vec.wv.syn0

word_size = word2vec.shape[1] #word2vec第二维长度为256，第一维为1056283

word2vec = np.concatenate([np.zeros((1, word_size)), word2vec])  #在第一行前加一行0

def tokenize(s):
    """
    分词
    """
    #return [i.word for i in pyhanlp.HanLP.segment(s)]
    return [i for i in jieba.cut(s)]



def sent2vec(S):
    """S格式：[[w1, w2]]
    """
    V = []

    for s in S:
        V.append([])
        for w in s:
            # if w in word2id:

              for _ in w:
                 V[(-1)].append(word2id.get(w, 0))


    V = seq_padding(V)


    V = word2vec[V]

    return V



def search_st_en(text_in, triple_list):  # 要修改找准确定位
    """
    找实体在原文中的起始位置
    """
    pre_subjects = {}
    # print(type(article['triple']))
    for triple in triple_list:
        subjects = []
        triple = triple.strip('(')
        triple = triple.strip(')')
        triple = triple.split(',')
        all_index1 = [substr.start() for substr in re.finditer(triple[0], text_in)]
        all_index2 = [substr.start() for substr in re.finditer(triple[2], text_in)]
        if len(all_index1) == 0:
            all_index1 = [substr.start() for substr in re.finditer(triple[0][0:3], text_in)]
        if len(all_index2) == 0:
            all_index2 = [substr.start() for substr in re.finditer(triple[2][0:3], text_in)]
        min1 = 0
        min2 = 0
        min = 10000000
        for value1 in all_index1:
            for value2 in all_index2:
                if abs(value2 - value1) < min:
                    min1 = value1
                    min2 = value2
                    min = abs(value2 - value1)

        pre_subjects[(min1, min1 + len(triple[0]))] = triple[0]
        pre_subjects[(min2, min2 + len(triple[2]))] = triple[2]

    return pre_subjects

def cut_sentences(sentence):
    para=re.sub('([。！？])([^”’])', r"\1\n\2", sentence)
    return para.split("\n")

def search_setence(text_in,triple_list):  # 找实体所在的句子,返回句子所在的序列

   for triple in triple_list:
       # print(triple)
       triple = triple.strip('(')
       triple = triple.strip(')')
       triple = triple.split(',')
       sentences=cut_sentences(text_in)
       for i,sent in enumerate(sentences):
           if(triple[0] in sent and triple[1] in sent and  triple[2] in sent):
               # print(sent)
               return i

       for i,sent in enumerate(sentences):
           if((triple[0] in sent and triple[1] in sent) or (triple[0] in sent and triple[2] in sent) \
                   or (triple[1] in sent and triple[2] in sent)):
               # print(sent)
               return i

       str=""
       str=str+triple[0]+triple[1]+triple[2]
       for i, sent in enumerate(sentences):
           if all([j in sent for j in str]):
               # print(sent)
               return i

'''
id2char={2: '摘', 3: '要', 4: '：', 5: '英', 6: '雄'.......}
char2id={'摘': 2, '要': 3, '：': 4, '英': 5, '雄': 6.......}
'''

id2char, char2id = json.load(open('data/all_chars_me.json'))



def seq_padding(X, padding=0):  # 让每条文本的长度相同，用0填充
    L = [len(x) for x in X]
    ML = max(L)
    return np.array([
        np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x
        for x in X
    ])


def isin_feature(text_a, text_b):  #标注text_b中的每个字符在text_a中出现的位置

    y = np.zeros(len(''.join(text_a)))

    text_b = set(text_b)

    i = 0
    for w in text_a:
        if w in text_b:

            for c in w:
                y[i] = 1
                i += 1

    return y






from keras.layers import *
from keras.models import Model
import keras.backend as K
from keras.callbacks import Callback
from keras.optimizers import Adam


class Attention(Layer):
    """多头注意力机制
    """
    def __init__(self, nb_head, size_per_head, **kwargs):
        self.nb_head = nb_head
        self.size_per_head = size_per_head
        self.out_dim = nb_head * size_per_head
        super(Attention, self).__init__(**kwargs)
    def build(self, input_shape):
        q_in_dim = input_shape[0][-1]
        k_in_dim = input_shape[1][-1]
        v_in_dim = input_shape[2][-1]
        self.q_kernel = self.add_weight(
            name='q_kernel',
            shape=(q_in_dim, self.out_dim),
            initializer='glorot_normal')
        self.k_kernel = self.add_weight(
            name='k_kernel',
            shape=(k_in_dim, self.out_dim),
            initializer='glorot_normal')
        self.v_kernel = self.add_weight(
            name='w_kernel',
            shape=(v_in_dim, self.out_dim),
            initializer='glorot_normal')
    def mask(self, x, mask, mode='mul'):
        if mask is None:
            return x
        for _ in range(K.ndim(x) - K.ndim(mask)):
            mask = K.expand_dims(mask, K.ndim(mask))
        if mode == 'mul':
            return x * mask
        else:
            return x - (1 - mask) * 1e10
    def call(self, inputs):
        q, k, v = inputs[:3]
        v_mask, q_mask = (None, None)
        if len(inputs) > 3:
            v_mask = inputs[3]
            if len(inputs) > 4:
                q_mask = inputs[4]
        # 线性变换
        qw = K.dot(q, self.q_kernel)
        kw = K.dot(k, self.k_kernel)
        vw = K.dot(v, self.v_kernel)
        # 形状变换
        qw = K.reshape(qw, (-1, K.shape(qw)[1], self.nb_head, self.size_per_head))
        kw = K.reshape(kw, (-1, K.shape(kw)[1], self.nb_head, self.size_per_head))
        vw = K.reshape(vw, (-1, K.shape(vw)[1], self.nb_head, self.size_per_head))
        # 维度置换
        qw = K.permute_dimensions(qw, (0, 2, 1, 3))
        kw = K.permute_dimensions(kw, (0, 2, 1, 3))
        vw = K.permute_dimensions(vw, (0, 2, 1, 3))
        # Attention
        a = K.batch_dot(qw, kw, [3, 3]) / self.size_per_head**0.5
        a = K.permute_dimensions(a, (0, 3, 2, 1))
        a = self.mask(a, v_mask, 'add')
        a = K.permute_dimensions(a, (0, 3, 2, 1))
        a = K.softmax(a)
        # 完成输出
        o = K.batch_dot(a, vw, [3, 2])
        o = K.permute_dimensions(o, (0, 2, 1, 3))
        o = K.reshape(o, (-1, K.shape(o)[1], self.out_dim))
        o = self.mask(o, q_mask, 'mul')
        return o
    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.out_dim)


def seq_maxpool(x):
    """seq是[None, seq_len, s_size]的格式，
    mask是[None, seq_len, 1]的格式，先除去mask部分，
    然后再做maxpooling。
    """
    seq, mask = x
    seq -= (1 - mask) * 1e10
    return K.max(seq, 1)


class MyBidirectional:
    """自己封装双向RNN，允许传入mask，保证对齐
    """
    def __init__(self, layer):
        self.forward_layer = layer.__class__.from_config(layer.get_config())
        self.backward_layer = layer.__class__.from_config(layer.get_config())
        self.forward_layer.name = 'forward_' + self.forward_layer.name
        self.backward_layer.name = 'backward_' + self.backward_layer.name
    def reverse_sequence(self, inputs):
        """这里的mask.shape是[batch_size, seq_len, 1]
        """
        x, mask = inputs
        seq_len = K.round(K.sum(mask, 1)[:, 0])
        seq_len = K.cast(seq_len, 'int32')
        return K.tf.reverse_sequence(x, seq_len, seq_dim=1)
    def __call__(self, inputs):
        x, mask = inputs
        x_forward = self.forward_layer(x)
        x_backward = Lambda(self.reverse_sequence)([x, mask])
        x_backward = self.backward_layer(x_backward)
        x_backward = Lambda(self.reverse_sequence)([x_backward, mask])
        x = Concatenate()([x_forward, x_backward])
        x = Lambda(lambda x: x[0] * x[1])([x, mask])
        return x


x1_in = Input(shape=(None, )) #shape定义输入数据的形状
x2_in = Input(shape=(None, ))
x1v_in = Input(shape=(None, word_size)) #输入第二维为256
x2v_in = Input(shape=(None, word_size))
s1_in = Input(shape=(None, ))
s2_in = Input(shape=(None, ))
pres1_in = Input(shape=(None, ))
pres2_in = Input(shape=(None, ))
y_in = Input(shape=(None, 1 + num_features)) #num_heature为2
t_in = Input(shape=(1, ))

x1, x2, x1v, x2v, s1, s2, pres1, pres2, y, t = (
    x1_in, x2_in, x1v_in, x2v_in, s1_in, s2_in, pres1_in, pres2_in, y_in, t_in
)


x1_mask = Lambda(lambda x: K.cast(K.greater(K.expand_dims(x, 2), 0), 'float32'))(x1) #对于每一个位置，只要值大于0，则为1，否则为0
x2_mask = Lambda(lambda x: K.cast(K.greater(K.expand_dims(x, 2), 0), 'float32'))(x2)

embedding = Embedding(len(id2char) + 2, char_size) #char_size为128

dense = Dense(char_size, use_bias=False)

x1 = embedding(x1)
x1v = dense(x1v)
x1 = Add()([x1, x1v])
x1 = Dropout(0.2)(x1)

pres1 = Lambda(lambda x: K.expand_dims(x, 2))(pres1)
pres2 = Lambda(lambda x: K.expand_dims(x, 2))(pres2)
x1 = Concatenate()([x1, pres1, pres2])
x1 = Lambda(lambda x: x[0] * x[1])([x1, x1_mask])


x1 = MyBidirectional(LSTM(char_size // 2, return_sequences=True))([x1, x1_mask])

h = Conv1D(char_size, 3, activation='relu', padding='same')(x1)
ps1 = Dense(1, activation='sigmoid')(h)
ps2 = Dense(1, activation='sigmoid')(h)
ps1 = Lambda(lambda x: x[0] * x[1])([ps1, pres1]) # 这样一乘，相当于只从最大匹配的结果中筛选实体
ps2 = Lambda(lambda x: x[0] * x[1])([ps2, pres2]) # 这样一乘，相当于只从最大匹配的结果中筛选实体

s_model = Model([x1_in, x1v_in, pres1_in, pres2_in], [ps1, ps2])


#实体链接
x1 = Concatenate()([x1, y])
x1 = MyBidirectional(LSTM(char_size // 2, return_sequences=True))([x1, x1_mask])
ys = Lambda(lambda x: K.sum(x[0] * x[1][..., :1], 1) / K.sum(x[1][..., :1], 1))([x1, y])

x2 = embedding(x2)
x2v = dense(x2v)
x2 = Add()([x2, x2v])
x2 = Dropout(0.2)(x2)
x2 = Lambda(lambda x: x[0] * x[1])([x2, x2_mask])
x2 = MyBidirectional(LSTM(char_size // 2, return_sequences=True))([x2, x2_mask])

x12 = Attention(8, 16)([x1, x2, x2, x2_mask, x1_mask])
x12 = Lambda(seq_maxpool)([x12, x1_mask])
x21 = Attention(8, 16)([x2, x1, x1, x1_mask, x2_mask])
x21 = Lambda(seq_maxpool)([x21, x2_mask])
x = Concatenate()([x12, x21, ys])
x = Dropout(0.2)(x)
x = Dense(char_size, activation='relu')(x)
pt = Dense(1, activation='sigmoid')(x)

t_model = Model([x1_in, x2_in, x1v_in, x2v_in, pres1_in, pres2_in, y_in], pt)



def extract_items(text_in,triple_list):

    text_words = tokenize(text_in)  #分词
    text_old = ''.join(text_words)
    text_in = text_old.lower()
    _x1 = [char2id.get(c, 1) for c in text_in]
    _x1 = np.array([_x1])#text_in的所有字符转换成id后的序列
    _x1v = sent2vec([text_words]) #text_words编码



    pre_subjects = search_st_en(text_in,triple_list)
    # print(pre_subjects)



    _pres1, _pres2 = np.zeros([1, len(text_in)]), np.zeros([1, len(text_in)])
    for j1, j2 in pre_subjects:
        _pres1[(0, j1)] = 1     #标注每一个实体的初始位置，在序列中置为1
        _pres2[(0, j2 - 1)] = 1  #标注每一个实体的结束位置，在序列中置为1
    _k1, _k2 = s_model.predict([_x1, _x1v, _pres1, _pres2])

    _k1, _k2 = _k1[0, :, 0], _k2[0, :, 0]
    _k1, _k2 = np.where(_k1 > 0.4)[0], np.where(_k2 > 0.4)[0]


    # _k1
    # [26  40  98 138 146 229]
    # _k2
    # [11  28  41 101 139 147]

    _subjects = []
    for i in _k1:
        j = _k2[(_k2 >= i)]
        if len(j) > 0:
            j = j[0]
            _subject = text_in[i:j + 1]
            _subjects.append((_subject, i, j + 1))


    if _subjects:
        R = []
        _X2, _X2V, _Y = [], [], []
        _S, _IDXS = [], {}
        _XS={}
        for _s in _subjects:
            _y1 = np.zeros(len(text_in))
            _y1[_s[1]: _s[2]] = 1 #对应一个0/1序列，将识别出来的实体标注为1，长度为text_in长度
            time.sleep(1)
            API_ENDPOINT = "http://shuyantech.com/api/cndbpedia/ment2ent?"
            params = {
                'q':_s[0],
                'apikey': 'fb7b030882c36f953a55ff9374dbacc8',
            }
            r = requests.get(API_ENDPOINT, params=params, verify=False)
            p1 = re.compile(r'[（,(](.*)[）,)]', re.S)  # 贪婪匹配
            des=[]
            for d in r.json()['ret']:
                n=re.findall(p1, d)
                if (len(n) == 0):
                    des.append(d)
                else:
                    des.append(n[0])
            _XS[_s]=des

            for i in des:


                _x2 = str(i)

                _x2_words = tokenize(_x2) #对_x2进行分词
                _x2 = ''.join(_x2_words)
                _y2 = isin_feature(text_in, _x2) #人工特征1：query的每个字是否在实体描述中出现过（对应一个0/1序列，序列长度为query长度）
                _y3 = isin_feature(text_words, _x2_words) #人工特征2：query和实体描述都分词，然后判断query每个词是否在实体描述中出现过（对应一个0/1序列，每个词对应的标记要重复词的长度那么多次，以保证得到通常长度的序列）
                _y = np.vstack([_y1, _y2, _y3]).T  # 将各向量纵向排列然后转置
                _x2 = [char2id.get(c, 1) for c in _x2] #将_x2中的每一个字符转换成id
                _X2.append(_x2) #记录所有实体属性描述转换成的id序列
                _X2V.append(_x2_words) #记录所有实体的属性描述分词后的所有词组集合
                _Y.append(_y) #记录所有人工特征
                _S.append(_s) #记录所有识别出来的实体
        if _X2:
            _X2 = seq_padding(_X2)
            _X2V = sent2vec(_X2V)
            _Y = seq_padding(_Y, np.zeros(1 + num_features))
            _X1 = np.repeat(_x1, len(_X2), 0)
            _X1V = np.repeat(_x1v, len(_X2), 0)
            _PRES1 = np.repeat(_pres1, len(_X2), 0)
            _PRES2 = np.repeat(_pres2, len(_X2), 0)
            scores = t_model.predict([_X1, _X2, _X1V, _X2V, _PRES1, _PRES2, _Y])[:, 0]

            for k, v in groupby(zip(_S, scores), key=lambda s: s[0]):



                vs = [j[1] for j in v]
                if np.max(vs) < 0.1:
                    continue

                link =_XS[k][np.argmax(vs)]
                R.append((text_old[k[1]:k[2]], k[1], link))

                # print(text_old[k[1]:k[2]]+"-----"+str(k[1])+"-----"+kbid)


        return R
    else:
        return []





def test():

    connect = pymongo.MongoClient(host='47.110.148.178', port=27017, username="root", password="8gd#729*1@5")
    mongo_newsdb = connect['SinaDataBase']
    collection = mongo_newsdb['Newsweibo_Test']

    articles = list(collection.find())


    for index,article in enumerate(articles):

        time.sleep(1)
        triple_des = {}
        text_in = ""
        for j in article['abstract']:
            text_in = text_in + j

        print("第%d篇文章----------------------------------"%(index+1))

        print(text_in)
        if text_in is "":
            triple_des={}
            collection.update_one({'_id': article['_id']}, {'$set': {
                'triple_des': triple_des
            }}, upsert=False)
            continue

        sentences=cut_sentences(text_in)
        for triple in article['triple']:
            triple_list=[]
            triple_list.append(triple)
            num=search_setence(text_in,triple_list)
            for md in set(extract_items(sentences[num],triple_list)):
                print(dict(zip(['mention', 'offset', 'kb_des'], [md[0], str(md[1]), md[2]])))
                triple_des[md[0]]=md[2]



        print(triple_des)
        collection.update_one({'_id': article['_id']}, {'$set': {
            'triple_des': triple_des
        }}, upsert=False)



if __name__ == '__main__':
   test()

