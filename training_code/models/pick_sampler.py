from collections import Counter
import sys
import torch
from tqdm import tqdm
import numpy as np
from scipy.sparse import csr_matrix


class LabelBalancedSampler:

    def __init__(self, A: np.array, labels: np.array, train_nid: np.array, multilabel: bool):
        self.A = A
        self.n = A.shape[0]
        # self.train_mask = train_mask
        self.train_n = train_nid.size
        self.D = self._calculate_D()
        self.A_hat = self._calculate_A_hat()
        self.train_labels = labels
        self.train_nid = train_nid
        self.multilabel = multilabel
        # print(train_nid,train_nid.size)
        # print(labels,labels.size)
        # self.train_labels = labels[train_nid]
        # self.train_index = np.where(train_mask)[0]
        if multilabel:
            count_freq = np.sum(self.train_labels, axis=0)
            none = len(self.train_labels)-len(np.nonzero(np.sum(self.train_labels, axis = 1))[0])
            self.label_frequency = count_freq
            self.label_frequency = np.append(self.label_frequency, none)
        else:
            count_freq = Counter(self.train_labels.tolist())
            self.label_frequency = np.zeros((max(count_freq.keys()) + 1), dtype=int)
            for i in count_freq: self.label_frequency[i] = count_freq[i]

    def _calculate_D(self) -> np.array:
        row = np.arange(0, self.n)
        col = np.arange(0, self.n)
        data = np.asarray(self.A.sum(axis=1)).flatten().astype(float)
        data[np.where(data == 0)] = 0.001
        D = csr_matrix((data, (row, col)), shape = (self.n, self.n)).tocoo()
        return D

    def _calculate_A_hat(self) -> np.array:
        row = np.arange(0, self.n)
        col = np.arange(0, self.n)
        data = 1 / np.sqrt(self.D.data)
        D_sqrt_inverse = csr_matrix((data, (row, col)), shape = (self.n, self.n)).tocoo()
        A_hat = (D_sqrt_inverse @ self.A @ D_sqrt_inverse).tocoo()
        return A_hat

    def _node_label_frequency(self):
        if self.multilabel:
            temp = self.train_labels[np.arange(self.train_n, dtype=int)]
            def func(x):
                y = np.nonzero(x)[0]
                if y.tolist() == []:
                    y = np.array([3])
                return np.min(self.label_frequency[y])
            label_freq = np.apply_along_axis(func, axis=1, arr=temp)
            return label_freq

        return self.label_frequency[self.train_labels[np.arange(self.train_n, dtype=int)]]

    # def calculate_P(self, node_idx: int) -> float:
    #     prob = np.linalg.norm(self.A_hat.getcol(node_idx).data, ord=2) / self._node_label_frequency(node_idx)
    
    def all_probabilities(self):
        col = self.A_hat.col
        data = self.A_hat.data ** 2
        
        final = np.zeros((self.n,))
        for a, b in zip(col, data): final[a] += b
        p = self._node_label_frequency()
        prob = np.sqrt(final)[self.train_nid] / p
        return prob / prob.sum()