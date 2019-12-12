import numpy as np
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
import pickle

from . import constants, IndexingComponent
from .. import utils
from ...new_voc import Voc

print = utils.get_printer(__name__)

CONST_DUMPED_FILE = 'dumped_file'


class BagWordIndexingComponent(IndexingComponent):
    default_hparams = dict(min_count=5)

    def __init__(self, hparams):
        IndexingComponent.__init__(self, hparams)

        if CONST_DUMPED_FILE in self.component_hparams:
            path_to_file = self.root_hparams[constants.GLOBAL]['tmp_dir'] + '/' + self.component_hparams[CONST_DUMPED_FILE]
            with open(path_to_file, 'rb') as i_f:
                self.vect = pickle.load(i_f)
            self.is_train = True
        else:
            self.vect = None
            self.is_train = False

    def train(self, docs, **kwargs):
        if not self.is_train:
            self.vect = Pipeline([('count_vect', CountVectorizer(min_df=self.component_hparams['min_count'])),
                                  ('tf_idf', TfidfTransformer())])
            self.vect.fit(docs)
        else:
            print('Already trained')
            print('vocab_size: %s' % len(self.vect.steps[0][1].get_feature_names()))

    def process(self, message, **kwargs):
        return self.vect.transform(message)

    def persist(self):
        if CONST_DUMPED_FILE not in self.component_hparams:
            self.component_hparams[CONST_DUMPED_FILE] = 'vect.pkl'
        path_to_file = self.root_hparams[constants.GLOBAL]['tmp_dir'] + '/' + self.component_hparams[CONST_DUMPED_FILE]
        with open(path_to_file, 'wb') as o_f:
            pickle.dump(self.vect, o_f)


class WordEmbeddingIndexingComponent(IndexingComponent):
    default_hparams = dict(size=300, window=10, min_count=5, workers=8, level='word')

    def __init__(self, hparams):
        IndexingComponent.__init__(self, hparams)
        if CONST_DUMPED_FILE in self.component_hparams:
            path_to_file = self.root_hparams[constants.GLOBAL]['tmp_dir'] + '/' + self.component_hparams[CONST_DUMPED_FILE]
            self.voc = Voc.load(path_to_file)
            self.is_train = True
            print('Load trained model successfully')
        else:
            self.voc = None
            self.is_train = False

    def process(self, messages, **kwargs):
        return self.voc.docs2idx(messages)

    def train(self, training_data, **kwargs):
        print('Fitting...')
        print('Current hparams: %s' % self.component_hparams)
        print('There are %s docs' % len(training_data))
        
        assert isinstance(training_data, list)
        assert isinstance(training_data[0], str)
        if self.component_hparams['level'] == 'word':
            all_docs = [doc.split() for doc in training_data]
            model = Word2Vec(all_docs, size=self.component_hparams['size'], window=self.component_hparams['window'],
                             min_count=self.component_hparams['min_count'], workers=self.component_hparams['workers'])
            wv = model.wv
            del model

            self.voc = Voc(tokenize_func=Voc.WORD_LV_TOK_FUNC, space_char=Voc.WORD_LV_SPACE_CHR)
            padding = '__padding__'
            oov = '__oov__'

            self.component_hparams['padding_idx'] = 0
            self.component_hparams['oov_idx'] = 1
            self.voc.build_from_tokens([padding, oov] + wv.index2word, padding_idx=self.component_hparams['padding_idx'],
                                       oov_idx=self.component_hparams['oov_idx'])

            padding_vector = np.zeros(self.component_hparams['size'])
            trained_weights = wv.syn0
            oov_vector = np.random.normal(loc=trained_weights.mean(), scale=trained_weights.std(), size=(self.component_hparams['size'],))
            self.voc.add_embedding_weights(np.concatenate((padding_vector.reshape(1, -1), oov_vector.reshape(1, -1), trained_weights), axis=0))

            self.voc.freeze()
            self.component_hparams['vocab_size'] = len(self.voc._index2word)
            self.describe(training_data, self.voc)

        elif self.component_hparams['level'] == 'char':
            raise NotImplemented('Only word is supported for now, while hparam.level=%s', self.component_hparams['level'])
        else:
            raise NotImplemented('Only word is supported for now, while hparam.level=%s', self.component_hparams['level'])

    def persist(self):
        if CONST_DUMPED_FILE not in self.component_hparams:
            self.component_hparams[CONST_DUMPED_FILE] = 'vect.pkl'
        path_to_file = self.root_hparams[constants.GLOBAL]['tmp_dir'] + '/' + self.component_hparams[CONST_DUMPED_FILE]
        self.voc.dump(path_to_file)

    def describe(self, docs, voc):
        print(f'vocab_size: {len(voc._index2word)}\t embedding_size: {voc.get_embedding_weights().shape[1]}')
        print('Some samples:')
        samples = [docs[2], docs[3], docs[5]]
        pos_samples = voc.idx2docs(voc.docs2idx(samples))
        print()
        for doc, pos_doc in zip(samples, pos_samples):
            print('\tOrigin: %s' % doc)
            print('\tPost: %s' % pos_doc)
        self.describe_coverage(docs, voc)
        print('-'*107 + '\n')

    def describe_coverage(self, docs, voc):
        docs = voc.docs2idx(docs)
        oov_count = 0
        word_count = 0
        for doc in docs:
            for tok in doc:
                if tok == voc._oov_idx:
                    oov_count += 1
                word_count += 1
        coverage_rate = (1 - oov_count / word_count)
        print('Coverage rate: %.2f' % coverage_rate)
        return coverage_rate

    def __str__(self):
        return str(self.component_hparams)

