from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import numpy as np
from util import log

rs = np.random.RandomState(123)


class Dataset(object):

    def __init__(self, ids, path, name='default', max_examples=None, is_train=True):
        self._ids = list(ids)
        self.name = name
        self.is_train = is_train
        self.path = path

        if max_examples is not None:
            self._ids = self._ids[:max_examples]

    def get_data(self, id):
        # preprocessing and data augmentation
        file = os.path.join(self.path, id)
        data = dict(np.load(file))
        img = data['image'].reshape((16, 160, 160, 1)) / 255.
        target = np.zeros(8, np.float32)
        target[data['target']] = 1
        meta_target = data['meta_target']
        return img, target, meta_target

    @property
    def ids(self):
        return self._ids

    def __len__(self):
        return len(self.ids)

    def __repr__(self):
        return 'Dataset (%s, %d examples)' % (
            self.name,
            len(self)
        )


def get_data_info():
    return np.array([16, 160, 160, 1, 8, 12])


def get_conv_info():
    return np.array([8, 16, 32, 64, 64])
    # return np.array([32, 32, 32, 32, 32])


def create_default_splits(path, is_train=True):
    ids = os.listdir(path)
    ids_dict = {
        'train': [],
        'val': [],
        'test': []
    }
    for idx in ids:
        ids_dict[idx.split('_')[2]].append(idx)

    # TODO: max examples
    dataset_train = Dataset(ids_dict['train'], path, name='train', is_train=False)
    dataset_test = Dataset(ids_dict['test'], path, name='test', is_train=False)
    return dataset_train, dataset_test


if __name__ == '__main__':
    datasets_train, datasets_test = create_default_splits('datasets/PGM/interpolation/')
    img, target = datasets_train.get_data(datasets_train.ids[0])
    print(img, target)
