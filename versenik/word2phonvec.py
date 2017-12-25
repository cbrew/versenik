from versenik.text2phon import to_phon
import gensim
import numpy as np
import scipy.spatial.distance as sp
import gensim.downloader as api


class UnknownDistanceMethod(Exception):
    pass


class Word2Phonvec(object):
    def __init__(self, filename):
        self.model = gensim.models.Word2Vec.load(filename)
        self.semantic_model = api.load('glove-wiki-gigaword-200')

    def to_vectors(self, word):
        toks = to_phon(word)
        return np.array([self.model.wv[p]
                         for tok in toks
                         for p in tok['phonetics'].split()])

    def to_vector(self, word, method="mean"):
        """
        Reduce a word to a single vector.
        """
        vectors = self.to_vectors(word)
        if len(vectors) == 0:
            return None
        if method == "mean":
            return vectors.mean(axis=0)
        elif method == "max":
            return vectors.max(axis=0)

    def distance(self, word1, word2, method="mean"):
        if method == "mean":
            return self._mean_distance(word1, word2)
        elif method == "max":
            return self._mean_distance(word1, word2, method="max")
        elif method == "viterbi":
            return self._viterbi_distance(word1, word2)
        else:
            raise UnknownDistanceMethod(method)

    def _viterbi_distance(self, word1, word2, **kwds):
        """
        Use the embedding vectors for a Viterbi
        match between the phones of word1 and
        the phones of word2.
        """
        toks1 = to_phon(word1)
        toks2 = to_phon(word2)
        phones1 = [p for tok in toks1 for p in tok['phonetics'].split()]
        phones2 = [p for tok in toks2 for p in tok['phonetics'].split()]
        return edit_distance(phones1, phones2, self.model), phones1, phones2

    def _mean_distance(self, word1, word2, method="mean", **kwds):
        vec1 = self.to_vector(word1, method=method)
        vec2 = self.to_vector(word2, method=method)
        if vec1 is None and vec2 is None:
            return 0.0
        elif vec1 is None or vec2 is None:
            return 1.0
        else:
            return sp.cosine(vec1, vec2)

    def suggest(self, word1, topn=100, method="mean"):
        """
        The suggestion mode uses phonemic embedding
        distance to select from the candidates provided by
        the semantic model.
        """
        if method == "mean" or method == "max":
            _distance = self._mean_distance
        elif method == "viterbi":
            _distance = self._viterbi_distance
        similar = [(_distance(word1, word2, method=method), semscore, word2)
                   for (word2, semscore)
                   in self.semantic_model.most_similar(word1, topn=topn)]

        return sorted(similar)


def edit_distance(phones1, phones2, model):
    def insdel_cost(phone):
        if phone == '-':
            return 5.0
        return 1.0

    def match_cost(phone1, phone2):
        if phone1 == "-" and phone2 != "-":
            return 1.9
        elif phone1 != "-" and phone2 == "-":
            return 1.9

        return sp.cosine(model.wv[phone1], model.wv[phone2])

    distance = np.zeros((len(phones1)+1, len(phones2)+1),
                        dtype=np.float32)
    distance[0, 0] = 0.0
    for i in range(len(phones1)):
        distance[i+1, 0] = distance[i, 0] + insdel_cost(phones1[i])
    for j in range(len(phones2)):
        distance[0, j+1] = distance[0, j] + insdel_cost(phones2[j])
    for i in range(len(phones1)):
        for j in range(len(phones2)):
            distance[i+1, j+1] = min(
                distance[i, j]+match_cost(phones1[i], phones2[j]),
                distance[i+1, j] + insdel_cost(phones1[i]),
                distance[i, j+1] + insdel_cost(phones2[j]))
    return distance[-1, -1]
