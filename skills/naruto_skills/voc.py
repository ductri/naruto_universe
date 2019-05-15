import json


class Voc:
    def __init__(self, name):
        # Default word tokens
        self.padding_tok_idx = 0
        self.oov_tok_idx = 1

        # Default settings, change them for char or word level
        self.tokenize_func = lambda x: [c for c in x]
        self.space_char = ''

        self.name = name
        self.trimmed = False
        self.init_default_value()

    def init_default_value(self):
        # Does not matter, just define how to render
        PADDING_CHAR = ''
        OOV_CHAR = '¶'

        # Does matter, be careful, do not change them
        PADDING_INDEX = 0
        OOV_INDEX = 1
        DEFAULT_INDEX2WORD = {PADDING_INDEX: PADDING_CHAR, OOV_INDEX: OOV_CHAR}
        DEFAULT_NUM_WORDS = 2
        self.word2index = {}
        self.word2count = {}
        self.num_words = DEFAULT_NUM_WORDS
        self.index2word = DEFAULT_INDEX2WORD

    def add_sentence(self, sentence):
        for word in self.tokenize_func(sentence):
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    # Remove words below a certain count threshold
    def trim(self, min_count):
        if self.trimmed:
            return
        self.trimmed = True

        keep_words = []

        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print('keep_words {} / {} = {:.4f}'.format(
            len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)
        ))

        # Reinitialize dictionaries

        self.init_default_value()

        for word in keep_words:
            self.add_word(word)

    def dump(self, path_file):
        with open(path_file, 'w') as o_f:
            json.dump(self.word2index, o_f)

    @staticmethod
    def load(f_json, name=''):
        voc = Voc(name)
        with open(f_json, 'r') as i_f:
            temp = json.load(i_f)
            voc.init_default_value()

            voc.word2index.update(temp)
            voc.index2word.update({v: k for k, v in voc.word2index.items()})
            voc.num_words = len(voc.index2word)

        return voc

    def docs2idx(self, docs, equal_length=-1):
        """

        :param docs:
        :param equal_length: -1 means keeping original length
        :param min_length: -1 means keeping original length
        :return:
        """
        docs = [self.tokenize_func(doc) for doc in docs]
        index_docs = [[self.word2index.get(token, self.oov_tok_idx) for token in doc] for doc in docs]
        index_docs = [self.__add_idx_padding(doc, equal_length) for doc in index_docs]
        return index_docs

    def __add_idx_padding(self, doc, length):
        """

        :param doc: list of idx
        :param length:
        :return:
        """
        return doc + (length - len(doc)) * [self.padding_tok_idx]

    def idx2docs(self, index_docs, is_skip_padding=True):
        if is_skip_padding is False:
            import pdb; pdb.set_trace()
        padding_token = self.index2word[self.padding_tok_idx] if not is_skip_padding else ''
        docs = [self.space_char.join(
                    [self.index2word[index_token] if index_token != self.padding_tok_idx else padding_token for index_token in
                     doc]).strip() for doc in index_docs]
        return docs


if __name__ == '__main__':
    voc = Voc('test')
    docs = ['hom nay toi di hoc', 'moi met qua di', 'hom nay la 1 ngay dep troi! :D']
    for doc in docs:
        voc.add_sentence(doc)
    assert len(voc.index2word) - len(voc.word2index) == 2
    print('Transform with %s vocab: %s' % (
        voc.num_words, voc.idx2docs(voc.docs2idx(['hom nay toi di hoc vao 1 ngay dep troi']))))

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
