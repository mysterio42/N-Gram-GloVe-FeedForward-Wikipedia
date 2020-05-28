import glob
import json
import os
import random
import string

import numpy as np
from scipy.sparse import csr_matrix, save_npz, load_npz
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


def dump_pmi(data, name):
    if not os.path.exists(EMBEDDING_DIR):
        os.makedirs(EMBEDDING_DIR.strip('/'))

    path = EMBEDDING_DIR + name + generate_model_name() + '.npz'

    save_npz(path, csr_matrix(data))
    print(f'{name} saved at {path}')


def load_pmi(path):
    return np.load(path)['arr_0']


def dump_word_encoding(name, data):
    if not os.path.exists(EMBEDDING_DIR):
        os.makedirs(EMBEDDING_DIR.strip('/'))

    path = EMBEDDING_DIR + name + '.json'
    with open(path, 'w') as f:
        json.dump(data, f)


def load_weight(path):
    npz = np.load(path)
    return npz['arr_0'], npz['arr_1']




def dump_weights(fn, W, U):
    if not os.path.exists(WEIGHTS_DIR):
        os.makedirs(WEIGHTS_DIR.strip('/'))
    path = WEIGHTS_DIR + fn

    arrays = [W, U.T]
    np.savez(path, *arrays)
    print(f'{fn} saved at {path}')


def dump_pmi_weight(fn, W):
    if not os.path.exists(WEIGHTS_DIR):
        os.makedirs(WEIGHTS_DIR.strip('/'))
    path = WEIGHTS_DIR + fn

    np.savez(path, W)
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


def top_neighs(word, word2idx, idx2word, W, top_k=10):
    V, D = W.shape

    if word not in word2idx:
        print(f'word: {word} not in the corpus')
        return
    word_encoding = word2idx[word]
    word_vector = W[word_encoding]

    distances = pairwise_distances(word_vector.reshape(1, D), W, metric='cosine').reshape(V)
    idx = distances.argsort()[:top_k]

    top_neighs = {}
    top_neighs['word'] = word
    top_neighs['neigh_dist'] = {idx2word[i]: distances[i] for i in idx if i != word_encoding}
    return top_neighs
