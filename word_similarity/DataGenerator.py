#! -*- coding:utf-8 -*-

'''
@Author:        ZM
@Date and Time: 2021/2/1 14:14
@File:          DataGenerator.py
'''

from sampler import BatchSampler

class BaseDataGenerator:
    def __init__(self, dataset, batch_size=1, shuffle=False, drop_last=False):
        super(BaseDataGenerator, self).__init__()
        self.dataset = dataset
        self.index_sampler = BatchSampler(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
        self._sampler_iter = iter(self.index_sampler)

    @property
    def sampler_iter(self):
        return self._sampler_iter

    def __len__(self):
        return len(self.index_sampler)

    def __iter__(self):
        return self

    def __next__(self):
        return self._next_data()

    def _next_index(self):
        try:
            index = next(self._sampler_iter)
        except StopIteration:
            self._sampler_iter = iter(self.index_sampler)
            index = next(self._sampler_iter)

        return index

    def _next_data(self):
        raise NotImplementedError