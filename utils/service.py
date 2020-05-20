import json

import numpy as np

from utils.network.glove import analogy, top_neighs
from utils.network.glove import load_weight


class Exps:

    def load_params(self, weight_path, embedding_path):
        W1, W2 = load_weight(weight_path)
        with open(embedding_path) as f:
            self._word2idx = json.load(f)
        self._idx2word = {i: w for w, i in self._word2idx.items()}
        self._W_concat = np.hstack([W1, W2.T])
        self._W_average = (W1 + W2.T) / 2
        del W1
        del W2

    def analogy(self, pos1, neg1, pos2):
        rets = {}
        rets['method'] = {}
        rets['method']['average'] = analogy(pos1, neg1, pos2, self._word2idx, self._idx2word, self._W_average)
        rets['method']['concat'] = analogy(pos1, neg1, pos2, self._word2idx, self._idx2word, self._W_concat)
        return rets

    def top_neigs(self, word,top_k=10):
        rets = {}
        rets['method'] = {}
        rets['method']['average'] = top_neighs(word, self._word2idx, self._idx2word, self._W_average,top_k)
        rets['method']['concat'] = top_neighs(word, self._word2idx, self._idx2word, self._W_concat,top_k)
        return rets
