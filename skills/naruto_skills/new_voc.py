import logging
import pickle
from collections import Counter


def _chr_lv_tok_func(x):
    """
    We need this because pickle cannot pickles lambda function without supporting of dill lib
    :return:
    """
    return [c for c in x]


class Voc:
    CHR_LV_TOK_FUNC = _chr_lv_tok_func
    CHR_LV_SPACE_CHR = ''
    WORD_LV_TOK_FUNC = str.split
    WORD_LV_SPACE_CHR = ' '
    PADDING_TOK = '__p__'
    OOV_TOK = '__o__'

    def __init__(self, tokenize_func=None, space_char=None):

        self.tokenize_func = tokenize_func
        self.space_char = space_char

        self.index2word = []
        self.__word2index = dict()
        self.padding_idx = -1
        self.oov_idx = -1
        self.embedding_weights = None

    def build_from_tokens(self, tokens, padding_idx, oov_idx):
        assert len(tokens) == len(set(tokens))
        self.index2word = tokens
        self.padding_idx = padding_idx
        self.oov_idx = oov_idx

    def add_embedding_weights(self, weights):
        self.embedding_weights = weights

    def __build_word2index(self):
        self.__word2index = {tok: idx for idx, tok in enumerate(self.index2word)}

    def dump(self, path_file):
        with open(path_file, 'wb') as o_f:
            pickle.dump({'index2word': self.index2word,
                         'tokenize_func': self.tokenize_func,
                         'space_char': self.space_char,
                         'padding_idx': self.padding_idx,
                         'oov_idx': self.oov_idx,
                         'embedding_weights': self.embedding_weights
                         }, o_f)

    @staticmethod
    def load(f_pkl):
        voc = Voc()
        with open(f_pkl, 'rb') as i_f:
            temp = pickle.load(i_f)
            voc.tokenize_func = temp['tokenize_func']
            voc.space_char = temp['space_char']
            voc.index2word = temp['index2word']
            voc.padding_idx = temp['padding_idx']
            voc.oov_idx = temp['oov_idx']
            voc.embedding_weights = temp['embedding_weights']
        return voc

    def docs2idx(self, docs, equal_length=-1):
        """

        :param docs:
        :param equal_length: -1 means keeping original length
        :param min_length: -1 means keeping original length
        :return:
        """
        docs = [self.tokenize_func(doc) for doc in docs]
        oov_index = self.oov_idx
        index_docs = [[self.word2index.get(token, oov_index) for token in doc] for doc in docs]
        index_docs = [self.__add_idx_padding(doc, equal_length) for doc in index_docs]
        return index_docs

    def __add_idx_padding(self, doc, length):
        """

        :param doc: list of idx
        :param length:
        :return:
        """
        padding_idx = self.padding_idx
        return doc + (length - len(doc)) * [padding_idx]

    def idx2docs(self, index_docs, is_skip_padding=True):
        padding_char = self.index2word[self.padding_idx] if not is_skip_padding else ''
        padding_idx = self.padding_idx

        docs = [self.space_char.join(
                    [self.index2word[index_token] if index_token != padding_idx else padding_char for index_token in
                     doc]).strip() for doc in index_docs]
        return docs

