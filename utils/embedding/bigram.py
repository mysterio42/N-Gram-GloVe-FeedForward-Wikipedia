import operator


def build_bigram_freq(embedding, start_idx, end_idx, smoothing):
    """
    Build Bi-Gram model based on Brown Corpus in json format
    :param embedding: Encoded Sentences
    :param start_idx: 'START' encoding
    :param end_idx: 'END' encoding
    :param smoothing: Smoothing parameter
    :return:
    """
    bigrams = {}
    for sentence in embedding:
        for i in range(len(sentence)):

            if i == 0:
                if start_idx not in bigrams:
                    bigrams[start_idx] = {}
                bigrams[start_idx][sentence[i]] = bigrams[start_idx].get(sentence[i], smoothing) + 1

            elif i == len(sentence) - 1:
                if sentence[i] not in bigrams:
                    bigrams[sentence[i]] = {}
                bigrams[sentence[i]][end_idx] = bigrams[sentence[i]].get(end_idx, smoothing) + 1
            else:
                if sentence[i - 1] not in bigrams:
                    bigrams[sentence[i - 1]] = {}
                bigrams[sentence[i - 1]][sentence[i]] = bigrams[sentence[i - 1]].get(sentence[i], smoothing) + 1
    return bigrams


def build_bigram_probs(corpus):
    """
    Figure out Bayes Rule based on Bi-Gram frequences


    :param corpus: Bi-Gram json
    :return: Bayes Inference
    """
    for last, couple in corpus.items():
        all_occurence = sum(couple.values())
        for current, freq in couple.items():
            corpus[last][current] = freq / all_occurence
    return corpus


def decode_bigrams(bigram_probs, idx2word):
    """
    Map encoded Bi-Grams to the real word


    :param bigram_probs: Bi-Gram json
    :param idx2word: Idx to Word map
    :return:
    """
    bigrams = {}

    for last, couple in bigram_probs.items():
        decode_last = idx2word[last]
        if decode_last not in bigrams:
            bigrams[decode_last] = {}
        for current, freq in couple.items():
            decode_current = idx2word[current]
            if decode_current not in bigrams[decode_last]:
                bigrams[decode_last][decode_current] = {}
            bigrams[decode_last][decode_current] = freq
    return bigrams


def sorted_bigrams(data):
    """
    Sort Bi-Gram based on his probabilities



    :param data: Bi-Gram json
    :return: Sorted Bi-Gram
    """
    def bigram_to_tuple(bigrams_proba: dict):
        helper = []
        for last, couple in bigrams_proba.items():
            for current, freq in couple.items():
                helper.append(((last, current), freq))
        return helper

    return sorted(bigram_to_tuple(data), key=operator.itemgetter(1), reverse=True)
