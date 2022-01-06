import os

import numpy as np


class ReprodLogger(object):
    def __init__(self):
        self._data = dict()

    @property
    def data(self):
        return self._data

    def add(self, key, val):
        self._data[key] = val

    def remove(self, key):
        if key in self._data:
            self._data.pop(key)
        else:
            print('{} is not in {}'.format(key, self._data.keys()))

    def clear(self):
        self._data.clear()

    def save(self, path):
        folder = os.path.dirname(path)
        if len(folder) >= 1:
            os.makedirs(folder, exist_ok=True)
        np.save(path, self._data)
