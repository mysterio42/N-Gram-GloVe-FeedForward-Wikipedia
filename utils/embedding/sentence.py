import operator

import numpy as np


def sentence_score(bigram_probs, encoded_sents, start_idx, end_idx):
    """
    Figure out sentence Cross-Entropy.

    :param bigram_probs: Bi-Gram probabilities json format
    :param encoded_sents: Encoded Sentence
    :param start_idx: 'START' encoding
    :param end_idx: 'END' encoding
    :return: Figure out sentence Cross-Entropy
    """
    score = 0
    try:
        for i in range(len(encoded_sents)):
            if i == 0:
                proba = bigram_probs[start_idx][encoded_sents[i]]
                score += proba * np.log(1 / proba)
            elif i == len(encoded_sents) - 1:
                proba = bigram_probs[encoded_sents[i]][end_idx]
                score += proba * np.log(1 / proba)
            else:
                proba = bigram_probs[encoded_sents[i - 1]][encoded_sents[i]]
                score += proba * np.log(1 / proba)
    except KeyError as s:
        score += np.log(1 / 0.01) * 0.01
        pass
    return score / (len(encoded_sents) + 1)


def sentence_inverse(idx2word, sentence):
    """

    :param sentence: Encoded Sentence
    :return: Decoded Sentence
    """
    return ' '.join(
        idx2word[embedded_word] for embedded_word in sentence if embedded_word in idx2word)


def next_pred(bigram_probs, word2idx, idx2word, last_word):
    """
    Figure out Bayes inference p(current_word|last_word).



    :param last_word: last word in sentence
    :return: Predicted word and probability
    """
    try:
        embedded_word = word2idx[last_word]
        encoded_current_word, proba = max(bigram_probs[embedded_word].items(), key=operator.itemgetter(1))
        current_word = idx2word[encoded_current_word]
        return current_word, proba
    except KeyError as e:
        pass


def nearest_neighbours(bigram_probs, word2idx, idx2word, last_word, top_n=5):
    """
    Figure out Bayes inference p(current_word|last_word).



    :param last_word: last word in sentence
    :return: Predicted word and probability
    """
    try:
        embedded_word = word2idx[last_word]

        encodedword_proba = sorted(bigram_probs[embedded_word].items(), key=operator.itemgetter(1), reverse=True)
        word_proba = [(idx2word[encword], proba) for encword, proba in encodedword_proba]

        if len(word_proba) >= top_n:
            return word_proba[:top_n]
        else:
            return word_proba

        # current_word = idx2word[encoded_current_word]
        # return current_word, proba
    except KeyError as e:
        pass


class FakeIdx:
    def __init__(self, dim, start_idx, end_idx):
        """"
        Generates random indexes to build random sentence

        """
        self.D = dim
        self.start_idx = start_idx
        self.end_idx = end_idx
        self._probas_init()

    def _probas_init(self):
        probas = np.ones(self.D)
        probas[self.start_idx] = 0
        probas[self.end_idx] = 0
        probas /= probas.sum()
        self.probas = probas
        del probas

    def __call__(self, window):
        return np.random.choice(self.D, size=window, p=self.probas)
