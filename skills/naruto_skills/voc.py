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

        self.word2count = Counter()

    def add_sentence(self, sentence):
        for word in self.tokenize_func(sentence):
            self.__add_word(word)

    def __add_word(self, word):
        self.word2count.update([word])

    def build_from_scratch(self, special_tokens=('', '__o__')):
        """

        :param special_tokens: in order: padding, oov
        :return:
        """
        vocabs = [k for k, v in self.word2count.most_common()]
        self.__validate_vocabs(vocabs, special_tokens)
        vocabs = list(special_tokens) + vocabs
        self.reindex(vocabs)

    def __validate_vocabs(self, vocabs, special_tokens):
        if special_tokens[0] in vocabs:
            logging.error('Token is same at padding_char: %s', special_tokens[0])
            raise Exception('Token is same at padding_char: %s' % special_tokens[0])
        if special_tokens[1] in vocabs:
            logging.error('Token is same at oov_char: %s', special_tokens[1])
            raise Exception('Token is same at oov_char: %s' % special_tokens[1])

    def reindex(self, vocabs):
        """

        :param vocabs: Without special tokens: padding, oov
        :return:
        """
        self.index2word = vocabs.copy()
        self.word2index = {tok: idx for idx, tok in enumerate(vocabs)}
        logging.info('Indexing vocabs successfully. Total vocabs: %s', len(self.index2word))

    def trim(self, min_count):
        original_len = len(self.word2count)
        vocabs = list(self.word2count.keys())
        for k in vocabs:
            if self.word2count[k] < min_count:
                del self.word2count[k]
        after_trimming_len = len(self.word2count)
        logging.info('keep_words {} / {} = {:.4f}'.format(after_trimming_len, original_len, after_trimming_len / original_len))

    def dump(self, path_file):
        with open(path_file, 'wb') as o_f:
            pickle.dump({'vocabs': self.index2word,
                         'tokenize_func': self.tokenize_func,
                         'space_char': self.space_char
                         }, o_f)

    @staticmethod
    def load(f_pkl):
        voc = Voc()
        with open(f_pkl, 'rb') as i_f:
            temp = pickle.load(i_f)
            voc.tokenize_func = temp['tokenize_func']
            voc.space_char = temp['space_char']
            voc.reindex(temp['vocabs'])
        return voc

    def docs2idx(self, docs, equal_length=-1):
        """

        :param docs:
        :param equal_length: -1 means keeping original length
        :param min_length: -1 means keeping original length
        :return:
        """
        docs = [self.tokenize_func(doc) for doc in docs]
        oov_index = 1
        index_docs = [[self.word2index.get(token, oov_index) for token in doc] for doc in docs]
        index_docs = [self.__add_idx_padding(doc, equal_length) for doc in index_docs]
        return index_docs

    def __add_idx_padding(self, doc, length):
        """

        :param doc: list of idx
        :param length:
        :return:
        """
        padding_idx = 0
        return doc + (length - len(doc)) * [padding_idx]

    def idx2docs(self, index_docs, is_skip_padding=True):
        padding_char = self.index2word[0] if not is_skip_padding else ''
        padding_idx = 0

        docs = [self.space_char.join(
                    [self.index2word[index_token] if index_token != padding_idx else padding_char for index_token in
                     doc]).strip() for doc in index_docs]
        return docs


if __name__ == '__main__':


    voc.dump('temp/test.json')
    new_voc = Voc.load('temp/test.json')
    print('Transform with %s vocab: %s' % (
        new_voc.num_words, new_voc.idx2docs(new_voc.docs2idx(['hom nay toi di hoc vao 1 ngay dep troi']))))

    voc = Voc('test')
    docs = ['hom nay toi di hoc', 'moi met qua di', 'hom nay la 1 ngay dep troi! :D']
    for doc in docs:
        voc.add_sentence(doc)
    assert len(voc.index2word) - len(voc.word2index) == 2
    print('Transform with %s vocab: %s' % (
        voc.num_words, voc.idx2docs(voc.docs2idx(['hom nay toi di hoc vao 1 ngay dep troi']))))

    voc.trim(2)
    voc.dump('temp/test.json')
    new_voc = Voc.load('temp/test.json')
    print('Transform with %s vocab: %s' % (
        new_voc.num_words, new_voc.idx2docs(new_voc.docs2idx(['hom nay toi di hoc vao 1 ngay dep troi']))))

    ###
    print('Word level')
    voc = Voc('test')
    voc.space_char = ' '
    voc.tokenize_func = str.split
    docs = ['hom nay toi di hoc', 'moi met qua di', 'hom nay la 1 ngay dep troi! :D']
    for doc in docs:
        voc.add_sentence(doc)
    assert len(voc.index2word) - len(voc.word2index) == 2
    print('Transform with %s vocab: %s' % (
        voc.num_words, voc.idx2docs(voc.docs2idx(['hom nay toi di hoc vao 1 ngay dep troi']))))
    voc.trim(2)
    voc.dump('temp/test.json')

    new_voc = Voc.load('temp/test.json')
    new_voc.space_char = ' '
    new_voc.tokenize_func = str.split
    print('Transform with %s vocab: %s' % (
        new_voc.num_words, new_voc.idx2docs(new_voc.docs2idx(['hom nay toi di hoc vao 1 ngay dep troi'], equal_length=50))))
    print('Transform with %s vocab: %s' % (
        new_voc.num_words,
        new_voc.idx2docs(new_voc.docs2idx(['hom nay toi di hoc vao 1 ngay dep troi'], equal_length=50), is_skip_padding=False)))
