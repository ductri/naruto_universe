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
        """
        list_tokens = [padding, oov] + [#all_tokens]
        voc = Voc(tokenize_func=Voc.WORD_LV_TOK_FUNC, space_char=Voc.WORD_LV_SPACE_CHR)
        voc.build_from_tokens(list_tokens, padding_idx=0, oov_idx=1)
        voc.freeze()
        voc.dump('voc')
        :param tokenize_func:
        :param space_char:
        """
        self.tokenize_func = tokenize_func
        self.space_char = space_char

        self.index2word = []
        self.padding_idx = -1
        self.oov_idx = -1

        self.__word2index = dict()
        self.__embedding_weights = None
        self.__is_freeze = False

    def build_from_tokens(self, tokens, padding_idx, oov_idx):
        """

        :param tokens: It must contain padding and oov
        :param padding_idx: Index of padding in `tokens`
        :param oov_idx: Index of padding in `oov`
        :return:
        """
        assert self.__is_freeze is False
        assert len(tokens) == len(set(tokens))
        self.index2word = tokens
        self.padding_idx = padding_idx
        self.oov_idx = oov_idx

    def add_embedding_weights(self, weights):
        assert self.__is_freeze is False
        assert len(self.index2word) == weights.shape[0]
        self.__embedding_weights = weights

    def get_embedding_weights(self):
        return self.__embedding_weights

    def freeze(self):
        assert self.__is_freeze is False
        self.__build_word2index()
        self.__is_freeze = True

    def __build_word2index(self):
        self.__word2index = {tok: idx for idx, tok in enumerate(self.index2word)}

    def dump(self, path_file):
        assert self.__is_freeze is True
        with open(path_file, 'wb') as o_f:
            pickle.dump({'index2word': self.index2word,
                         'tokenize_func': self.tokenize_func,
                         'space_char': self.space_char,
                         'padding_idx': self.padding_idx,
                         'oov_idx': self.oov_idx,
                         'embedding_weights': self.__embedding_weights
                         }, o_f)

    @staticmethod
    def load(f_pkl):
        voc = Voc()
        with open(f_pkl, 'rb') as i_f:
            temp = pickle.load(i_f)
            voc.tokenize_func = temp['tokenize_func']
            voc.space_char = temp['space_char']
            voc.padding_idx = temp['padding_idx']
            voc.oov_idx = temp['oov_idx']
            voc.index2word = temp['index2word']
            if temp['embedding_weights'] is not None:
                voc.add_embedding_weights(temp['embedding_weights'])
            voc.freeze()
        return voc

    def docs2idx(self, docs, equal_length=-1):
        """

        :param docs:
        :param equal_length: -1 means keeping original length
        :param min_length: -1 means keeping original length
        :return:
        """
        assert self.__is_freeze is True
        docs = [self.tokenize_func(doc) for doc in docs]
        oov_index = self.oov_idx
        index_docs = [[self.__word2index.get(token, oov_index) for token in doc] for doc in docs]
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
        assert self.__is_freeze is True
        padding_char = self.index2word[self.padding_idx] if not is_skip_padding else ''
        padding_idx = self.padding_idx

        docs = [self.space_char.join(
                    [self.index2word[index_token] if index_token != padding_idx else padding_char for index_token in
                     doc]).strip() for doc in index_docs]
        return docs


class VocHelper:
    def __init__(self):
        self.word2count = Counter()

    def add_docs(self, preprocessed_docs):
        docs = [doc.split() for doc in preprocessed_docs]
        tokens = [tok for doc in docs for tok in doc]
        for tok in tokens:
            self.word2count.update(tok)
