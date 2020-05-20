import glob
import os
import random
import string

import numpy as np
from sklearn.metrics.pairwise import pairwise_distances

EMBEDDING_DIR = 'embeddings/'
WEIGHTS_DIR = 'weights/'


def generate_model_name(size=5):
    """

    :param size: name length
    :return: random lowercase and digits of length size
    """
    letters = string.ascii_lowercase + string.digits
    return ''.join(random.choice(letters) for _ in range(size))


def latest_modified_weight():
    """
    returns latest trained weight
    :return: model weight trained the last time
    """
    weight_files = glob.glob(EMBEDDING_DIR + '*')
    latest = max(weight_files, key=os.path.getctime)
    return latest


def dump_embedding(data, name):
    if not os.path.exists(EMBEDDING_DIR):
        os.makedirs(EMBEDDING_DIR.strip('/'))

    path = EMBEDDING_DIR + name + generate_model_name() + '.npz'
    np.savez(path, data)
    print(f'{name} saved at {path}')


def load_weight(path):
    return np.load(path)


def dump_weights(fn, W, U):
    if not os.path.exists(WEIGHTS_DIR):
        os.makedirs(WEIGHTS_DIR.strip('/'))
    path = WEIGHTS_DIR + fn + 'npz'

    arrays = [W, U.T]
    np.savez(path, *arrays)
    print(f'{fn} saved at {path}')


def analogy(pos1, neg1, pos2, word2idx, idx2word, W):
    V, D = W.shape
    for w in (pos1, neg1, pos2):
        if w not in word2idx:
            print(f'word: {w} not in the corpus')
            return

    p1, n1, n2 = W[word2idx[pos1]], W[word2idx[neg1]], W[word2idx[pos2]]

    vec = p1 - n1 + n2

    distances = pairwise_distances(vec.reshape(1, D), W, metric='cosine').reshape(V)
    idx = distances.argsort()[:10]

    best_idx = -1
    for i in idx:
        if i not in [word2idx[w] for w in (pos1, neg1, pos2)]:
            best_idx = i
            break

    equation = f'{pos1}-{neg1}+{pos2}'
    solution = idx2word[best_idx]
    rets = {'equation': equation,
            'solution': solution}
    return rets
