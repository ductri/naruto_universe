import logging

import numpy as np
from gensim.models import Word2Vec


class WordEmbedding:
    def __init__(self, preprocessed_docs, min_freq, embedding_size, worker=4, **params):
        """

        :param preprocessed_docs:
        :param min_freq:
        :param embedding_size:
        :param params:
        """
        docs = [doc.split() for doc in preprocessed_docs]

        model = Word2Vec(docs, size=embedding_size, window=10, min_count=min_freq, workers=worker, **params)
        self.wv = model.wv

    def save_it(self, path_for_weight, path_for_vocab):
        weights = self.get_weight()
        np.save(path_for_weight, weights)
        logging.info('Dumped word embedding weights with shape %s to %s.npy', weights.shape, path_for_weight)
        with open(path_for_vocab, 'wt', encoding='utf-8') as output_f:
            vocabs = self.get_vocab()
            vocabs = [tok + '\n' for tok in vocabs]
            output_f.writelines(vocabs)
        logging.info('Dumped %s vocabs to %s', len(vocabs), path_for_vocab)

    @staticmethod
    def load_it(path_for_weight, path_for_vocab):
        raise NotImplementedError('It has not been necessary now')

    def add_vocab(self, tokens):
        self.wv.add(tokens, np.random.normal(loc=self.get_weight().mean(), scale=self.get_weight().std(),
                                             size=(len(tokens), self.wv.syn0.shape[1])))

    def get_vocab(self):
        return self.wv.index2word

    def get_weight(self):
        return self.wv.syn0

