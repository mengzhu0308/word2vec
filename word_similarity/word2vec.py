#! -*- coding:utf-8 -*-

'''
@Author:        ZM
@Date and Time: 2021/3/4 5:50
@File:          word2vec.py
'''

from keras import backend as K
from keras.layers import *

def cbow(x, vocub_size, word_dim, input_length):
    x = Embedding(vocub_size, word_dim, input_length=input_length)(x)
    x = Lambda(lambda arg: K.sum(arg, axis=1))(x)
    x = Dense(vocub_size, activation='softmax')(x)
    return x

def skip_gram(x, vocub_size, word_dim, window_size):
    x = Embedding(vocub_size, word_dim, input_length=1)(x)
    x = Lambda(lambda arg: arg[:, 0, :])(x)
    return [Dense(vocub_size, activation='softmax')(x) for _ in range(window_size * 2)]
