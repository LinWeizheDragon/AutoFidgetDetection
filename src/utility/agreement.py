from sklearn.metrics import cohen_kappa_score
import numpy as np


class Agreement():

    def __init__(self, data1, data2):
        self.data1 = data1
        self.data2 = data2

    def get_score(self):
        overlap_array = np.zeros_like(self.data1)
        print('overlap:')
        overlap_array[(self.data1 == 1) & (self.data2 == 1)] = 1
        print(
            np.count_nonzero(overlap_array) / min(
                np.count_nonzero(self.data1),
                np.count_nonzero(self.data2)
            )
        )
        print('total on 1')
        print(
            np.count_nonzero(self.data1)
        )
        print('total on 2')
        print(
            np.count_nonzero(self.data2)
        )
        return np.count_nonzero(overlap_array) / np.count_nonzero(self.data1)


