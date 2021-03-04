#! -*- coding:utf-8 -*-

'''
@Author:        ZM
@Date and Time: 2021/3/4 5:50
@File:          train_eval_cbow.py
'''

import math
import pickle
import numpy as np
from gensim import corpora
from pprint import pprint
from keras import Model, Input
from keras.callbacks import Callback
from keras.optimizers import Adam

from Dataset import Dataset
from DataGenerator import BaseDataGenerator
from word2vec import cbow

if __name__ == '__main__':
    word_dim = 128
    window_size = 7
    min_df = 10
    subsample_t = 1e-5
    batch_size = 2000
    epochs = 600
    init_lr = 0.2

    with open('sentences.pkl', 'rb') as f:
        sentences = pickle.load(f)

    dictionary = corpora.Dictionary(sentences)
    dfs = {dictionary[i]: j for i, j in dictionary.dfs.items() if j >= min_df}
    id2token = ['[PAD]', '[UNK]'] + list(dfs.keys())
    token2id = {j: i for i, j in enumerate(id2token)}
    vocub_size = len(id2token)

    total_df = sum(dictionary.dfs.values())
    subsamples = {i: j / total_df for i, j in dfs.items() if j / total_df > subsample_t}
    subsamples = {i: subsample_t / j + (subsample_t / j) ** 0.5 for i, j in subsamples.items()}
    subsamples = {token2id[i]: j for i, j in subsamples.items() if j < 1.}

    class DataGenerator(BaseDataGenerator):
        def __init__(self, dataset, **kwargs):
            super(DataGenerator, self).__init__(dataset, **kwargs)

        def _next_data(self):
            index = self._next_index()
            batch_x, batch_y = [], []
            for idx in index:
                sent = self.dataset[idx]
                rd = np.random.random(len(sent))
                for i in range(window_size, len(sent) - window_size):
                    if sent[i] in subsamples.keys() and rd[i] > subsamples[sent[i]]:
                        continue
                    batch_x.append(np.concatenate([sent[i - window_size: i], sent[i + 1: i + 1 + window_size]]))
                    batch_y.append(sent[i])

            batch_x, batch_y = np.array(batch_x), np.array(batch_y)

            return batch_x, batch_y

    def sent2id(sent, UNK=1):
        target_sent = []
        for w in sent:
            try:
                target_sent.append(token2id[w])
            except KeyError:
                target_sent.append(UNK)

        return np.array(target_sent, dtype='int32')

    def sequence_padding(inputs, padding=0):
        pad_width = (window_size, window_size)
        outputs = []
        for x in inputs:
            x = np.pad(x, pad_width, 'constant', constant_values=padding)
            outputs.append(x)

        return outputs

    '''train
    '''
    x = [sent2id(sent) for sent in sentences]
    x = sequence_padding(x)

    dataset = Dataset(x)
    gen = DataGenerator(dataset, batch_size=batch_size, shuffle=True)

    x_in = Input(shape=(window_size * 2, ), dtype='int32')
    y_out = cbow(x_in, vocub_size, word_dim, window_size * 2)
    model = Model(x_in, y_out)
    optimizer = Adam(learning_rate=init_lr)
    model.compile(optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    class Evaluator(Callback):
        def __init__(self):
            super(Evaluator, self).__init__()
            self.best_loss = math.inf

        def on_epoch_end(self, epoch, logs=None):
            if logs['loss'] < self.best_loss:
                model.save_weights('cbow_best_wts.weights')

    evaluator = Evaluator()

    model.fit_generator(gen,
                        steps_per_epoch=len(gen),
                        epochs=epochs,
                        callbacks=[evaluator],
                        shuffle=False,
                        initial_epoch=0)

    '''eval
    '''
    model.load_weights('cbow_best_wts.weights', by_name=True)
    embedding_wts = model.get_weights()[0]

    # cos相似
    norm_embedding_wts = embedding_wts / np.sqrt(np.sum(np.square(embedding_wts), axis=1, keepdims=True))
    def cos_similarity(word):
        vector = norm_embedding_wts[token2id[word]]
        sims = np.einsum('mn, n->m', norm_embedding_wts, vector)
        sort = np.argsort(sims)[::-1]
        return [(id2token[i], sims[i]) for i in sort[:6]]

    pprint(cos_similarity('people'))