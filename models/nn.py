from datetime import datetime

import numpy as np

from utils.network.glove import dump_embedding
from utils.network.glove import dump_weights
from utils.plot import plot_costs


class Glove:
    def __init__(self, embedding, hidden_dim, n_gram):
        self.context_sz = n_gram
        self.embedding = embedding
        self.D = hidden_dim
        self.V = len(self.embedding.word2idx)

    def build(self):
        """


        build co-occurence matrix

        :return:
        """
        self.build_X()
        self.weighting(xmax=100, alpha=0.75)
        self.init_params()

    def build_X(self):
        t0 = datetime.now()

        co_occurance = np.zeros((self.V, self.V))
        sen_len = len(self.embedding.encoded_sentences)

        print(f"number of sentences to process {sen_len}")
        it = 0
        for sentence in self.embedding.encoded_sentences:
            it += 1
            if it % 10000 == 0:
                print(f'processed {it}/{sen_len}')
            n = len(sentence)
            if n < 2:
                continue

            for i in range(n):
                wi = sentence[i]

                start = max(0, i - self.context_sz)
                end = min(n, i + self.context_sz)

                if i - self.context_sz < 0:
                    points = 1.0 / (i + 1)
                    co_occurance[wi, 0] += points
                    co_occurance[0, wi] += points

                if i + self.context_sz > n:
                    points = 1.0 / (n - 1)
                    co_occurance[wi, 1] += points
                    co_occurance[1, wi] += points

                for j in range(start, i):
                    wj = sentence[j]
                    points = 1.0 / (i - j)
                    co_occurance[wi, wj] += points
                    co_occurance[wj, wi] += points

                for j in range(i + 1, end):
                    wj = sentence[j]
                    points = 1.0 / (j - i)
                    co_occurance[wi, wj] += points
                    co_occurance[wj, wi] += points

        self.X = co_occurance
        print("time to build co-occurrence matrix:", (datetime.now() - t0))
        dump_embedding(co_occurance, 'glove_model_50')

    def weighting(self, xmax, alpha):
        print("max in X:", np.max(self.X))

        fX = np.zeros((self.V, self.V))
        fX[self.X < xmax] = (self.X[self.X < xmax] / float(xmax)) ** alpha
        fX[self.X >= xmax] = 1

        self.fX = fX
        self.target = np.log(self.X + 1)

        print("max in f(X):", np.max(self.fX))

    def init_params(self):
        self.W = np.random.randn(self.V, self.D) / np.sqrt(self.V + self.D)
        self.b = np.zeros(self.V)
        self.U = np.random.randn(self.V, self.D) / np.sqrt(self.V + self.D)
        self.c = np.zeros(self.V)
        self.mu = self.target.mean()

    def train(self, epochs=10, reg=0.1, lr=1e-4, gd=False):
        costs = []
        for epoch in range(epochs):
            delta = self.W.dot(self.U.T) + self.b.reshape(self.V, 1) + self.c.reshape(1, self.V) + self.mu - self.target
            cost = (self.fX * delta * delta).sum()
            costs.append(cost)
            print(f'epoch: {epoch}/{epochs} cost: {cost}')

            if gd:
                for i in range(self.V):
                    self.W[i] -= lr * (self.fX[i, :] * delta[i, :]).dot(self.U)
                self.W -= lr * reg * self.W

                for i in range(self.V):
                    self.b[i] -= lr * self.fX[i, :].dot(delta[i, :])

                for j in range(self.V):
                    self.U[j] -= lr * (self.fX[:, j] * delta[:, j]).dot(self.W)
                self.U -= lr * reg * self.U

                for j in range(self.V):
                    self.c[j] -= lr * self.fX[:, j].dot(delta[:, j])
            else:

                t0 = datetime.now()
                for i in range(self.V):
                    matrix = reg * np.eye(self.D) + (self.fX[i, :] * self.U.T).dot(self.U)
                    vector = (self.fX[i, :] * (self.target[i, :] - self.b[i] - self.c - self.mu)).dot(self.U)
                    self.W[i] = np.linalg.solve(matrix, vector)
                print(f'fast way took: {datetime.now() - t0}')

                for i in range(self.V):
                    denominator = self.fX[i, :].sum() + reg
                    numerator = self.fX[i, :].dot(self.target[i, :] - self.W[i].dot(self.U.T) - self.c - self.mu)

                    self.b[i] = numerator / denominator

                for j in range(self.V):
                    matrix = reg * np.eye(self.D) + (self.fX[:, j] * self.W.T).dot(self.W)

                    vector = (self.fX[:, j] * (self.target[:, j] - self.b - self.c[j] - self.mu)).dot(self.W)

                    self.U[j] = np.linalg.solve(matrix, vector)

                for j in range(self.V):
                    denominator = self.fX[:, j].sum() + reg
                    numerator = self.fX[:, j].dot(self.target[:, j] - self.W.dot(self.U[j]) - self.b - self.mu)
                    self.c[j] = numerator / denominator

        nm = 'GD' if gd else 'ALS'
        dump_weights(f'glove_model_{nm}50', self.W, self.U)
        plot_costs(costs, nm)
