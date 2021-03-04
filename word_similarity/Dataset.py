'''
@Author:        ZM
@Date and Time: 2019/10/8 6:28
@File:          Dataset.py
'''

class Dataset:
    def __init__(self, x, y=None, transform=None, y_transform=None):
        self.x = x
        self.y = y
        self.transform = transform
        self.y_transform = y_transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item):
        x = self.x[item]
        if self.transform is not None:
            x = self.transform(x)

        if self.y is not None:
            y = self.y[item]
            if self.y_transform is not None:
                y = self.y_transform(y)

            return x, y

        return x