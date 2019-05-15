import os
import logging
import argparse

import pandas as pd


class TextTransformer:
    transformer = None
    PADDING = '__padding__'
    OOV = '__oov__'
    START = '__s__'
    END = '__e__'
    TOKENIZE_FUNC = str.split
    SPACE_CHAR = ' '

    def __init__(self, vocabs, padding_token=None, oov_token=None, tokenize_func=None, space_char=None, other_special_tokens=()):
        self.vocabs = vocabs

        self.space_char = TextTransformer.SPACE_CHAR if space_char is None else space_char
        self.tokenize_func = TextTransformer.TOKENIZE_FUNC if tokenize_func is None else tokenize_func
        self.padding_token = TextTransformer.PADDING if padding_token is None else padding_token
        if self.padding_token not in self.vocabs:
            logging.warning('Missing PADDING token in vocabs. Auto using "%s" as PADDING', self.padding_token)
            self.vocabs.append(self.padding_token)

        self.oov_token = TextTransformer.OOV if oov_token is None else oov_token
        if self.oov_token not in self.vocabs:
            logging.warning('Missing OOV token in vocabs. Auto using "%s" as OOV', self.oov_token)
            self.vocabs.append(self.oov_token)

        for special_tok in other_special_tokens:
            if special_tok not in self.vocabs:
                logging.warning('Missing %s token in vocabs. Appended', special_tok)
                self.vocabs.append(special_tok)

        logging.info('Init vocab successfully. It contains %s words', len(self.vocabs))

        self.inverse_vocabs = {token: index for index, token in enumerate(self.vocabs)}

    def docs_to_index(self, docs, max_length=-1, min_length=-1):
        """

        :param docs:
        :param max_length: -1 means keeping original length
        :param min_length: -1 means keeping original length
        :return:
        """
        docs = [self.tokenize_func(doc) for doc in docs]
        if max_length != -1:
            docs = [(doc + [self.padding_token]*(min_length - len(doc)))[:max_length] for doc in docs]
        else:
            docs = [(doc + [self.padding_token] * (min_length - len(doc))) for doc in docs]

        oov_index = self.inverse_vocabs[self.oov_token]
        index_docs = [[self.inverse_vocabs.get(token, oov_index) for token in doc] for doc in docs]
        return index_docs

    def index_to_docs(self, index_docs, is_skip_padding=True):
        if is_skip_padding:
            return [self.space_char.join([self.vocabs[index_token] for index_token in doc
                              if self.vocabs[index_token] != self.padding_token]) for doc in index_docs]
        else:
            return [self.space_char.join([self.vocabs[token] for token in doc]) for doc in index_docs]

    @staticmethod
    def get_instance(vocab_file, padding_token=None, oov_token=None, tokenize_func=None, space_char=None, other_special_tokens=()):
        # TODO
        TextTransformer.transformer = None

        if TextTransformer.transformer is None:
            logging.info('Loading vocab from %s', vocab_file)
            with open(vocab_file, 'rt', encoding='utf-8') as input_f:
                vocabs = input_f.readlines()
                vocabs = [vocab.strip() for vocab in vocabs]
                vocabs = [vocab for vocab in vocabs if vocab != '']
            TextTransformer.transformer = TextTransformer(vocabs, padding_token, oov_token, tokenize_func, space_char, other_special_tokens)
        return TextTransformer.transformer

    def save_vocab(self, path):
        dir = os.path.dirname(path)
        if not os.path.exists(dir):
            os.makedirs(dir)
        with open(path, 'wt', encoding='utf-8') as output_f:
            output_f.writelines([line + '\n' for line in self.vocabs])
        logging.info('Dumped %s vocabs to %s', len(self.vocabs), path)
