from versenik.text2phon import to_phon
import gensim
import numpy as np
import scipy.spatial.distance as sp
import gensim.downloader as api


class Word2Phonvec(object):
    def __init__(self, filename):
        self.model = gensim.models.Word2Vec.load(filename)
        self.semantic_model = api.load('glove-wiki-gigaword-200')

    def to_vectors(self, word):
        toks = to_phon(word)
        return np.array([self.model.wv[p]
                         for tok in toks
                         for p in tok['phonetics'].split()])

    def to_vector(self, word):
        vectors = self.to_vectors(word)
        if len(vectors) == 0:
            return None
        return vectors.mean(axis=0)

    def distance(self, word1, word2):
        if isinstance(word1, str):
            vec1 = self.to_vector(word1)
        else:
            vec1 = word1

        return sp.cosine(vec1, self.to_vector(word2))

    def suggest(self, word, topn=100):
        """
        The suggestion mode uses phonemic embedding
        distance to select from the candidates provided by
        the semantic model.
        """
        vec = self.to_vector(word)
        words = self.semantic_model.most_similar(word,
                                                 topn=topn)
        vecs = []
        for (w, semscore) in words:
            v = self.to_vector(w)
            if v is not None:
                vecs.append((w, v, semscore))

        return sorted([
                (1-sp.cosine(vec, v), semscore, w)
                for (w, v, semscore) in vecs
                if v is not None],
                      reverse=True)
