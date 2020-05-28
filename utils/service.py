import json

import numpy as np
from sklearn.manifold import TSNE

from utils.network.glove import analogy, top_neighs
from utils.network.glove import load_weight,load_pmi
from utils.plot import plot_countries


class Exps:

    def load_pmi_params(self, weight_path, embedding_path):
        self.W = load_pmi(weight_path)
        with open(embedding_path) as f:
            self._word2idx = json.load(f)
        self._idx2word = {i: w for w, i in self._word2idx.items()}

    def load_params(self, weight_path, embedding_path):
        W1, W2 = load_weight(weight_path)
        with open(embedding_path) as f:
            self._word2idx = json.load(f)
        self._idx2word = {i: w for w, i in self._word2idx.items()}
        self._W_concat = np.hstack([W1, W2.T])
        self._W_average = (W1 + W2.T) / 2
        del W1
        del W2

    def analogy_pmi(self, pos1, neg1, pos2):
        rets = {}
        rets['method'] = {}
        rets['method']['whole'] = analogy(pos1, neg1, pos2, self._word2idx, self._idx2word, self.W)
        return rets
    def analogy(self, pos1, neg1, pos2):
        rets = {}
        rets['method'] = {}
        rets['method']['average'] = analogy(pos1, neg1, pos2, self._word2idx, self._idx2word, self._W_average)
        rets['method']['concat'] = analogy(pos1, neg1, pos2, self._word2idx, self._idx2word, self._W_concat)
        return rets

    def top_neigs(self, word, top_k=10):
        rets = {}
        rets['method'] = {}
        rets['method']['average'] = top_neighs(word, self._word2idx, self._idx2word, self._W_average, top_k)
        rets['method']['concat'] = top_neighs(word, self._word2idx, self._idx2word, self._W_concat, top_k)
        return rets

    def top_neigs_pmi(self, word, top_k=10):
        rets = {}
        rets['method'] = {}
        rets['method']['first'] = top_neighs(word, self._word2idx, self._idx2word, self.W, top_k)
        return rets

    def visualize_countries(self, words):
        idx = [self._word2idx[w] for w in words]
        plot_countries(TSNE().fit_transform(self._W_average)[idx], words, 'average')
        plot_countries(TSNE().fit_transform(self._W_concat)[idx], words, 'concatenate')

    def visualize_countries_pmi(self, words):
        idx = [self._word2idx[w] for w in words]
        plot_countries(TSNE().fit_transform(self.W)[idx], words, 'first')
