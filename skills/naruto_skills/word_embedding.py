import logging

import numpy as np
from gensim.models import Word2Vec
from gensim.models.keyedvectors import Word2VecKeyedVectors
import pandas as pd


class WordEmbedding:
    def __init__(self, preprocessed_csv_file, min_freq, embedding_size, interesting_column, **params):
        """

        :param preprocessed_csv_file:
        :param min_freq:
        :param embedding_size:
        :param interesting_column: Text column
        :param params:
        """
        df = pd.read_csv(preprocessed_csv_file, lineterminator='\n')
        no_null_rows = df[interesting_column].isnull().sum()
        if no_null_rows > 0:
            logging.warning('Column %s has %s empty instances. We are going to remove them all', interesting_column,
                            no_null_rows)
            df.dropna(subset=[interesting_column], inplace=True)
        docs = [[token for token in doc.split()] for doc in df[interesting_column]]

        model = Word2Vec(docs, size=embedding_size, window=10, min_count=min_freq, workers=4, **params)
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

    def save_json(self, path_for_weight, path_for_vocab):
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
        self.wv.add(tokens, np.random.normal(loc=0., scale=0.001, size=(len(tokens), self.wv.syn0.shape[1])))

    def get_vocab(self):
        return self.wv.index2word

    def get_weight(self):
        return self.wv.syn0

