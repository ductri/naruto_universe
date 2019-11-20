import numpy as np
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
import pickle

from naruto_skills.quick_training import constants, utils
from naruto_skills.new_voc import Voc

print = utils.get_printer(__name__)


class IndexingTransform:
    def __init__(self, hparams):
        self.root_hparams = hparams
        self.module_hparams = hparams[constants.INDEXING_TRANSFORM]

    def fit(self, docs):
        raise NotImplemented()

    def transform(self, docs):
        raise NotImplemented

    def fit_and_transform(self, docs):
        raise NotImplemented


class BagWordIndexingTransform(IndexingTransform):
    def __init__(self, hparams):
        IndexingTransform.__init__(self, hparams)
        default_hparams = dict(min_count=5)
        default_hparams.update(self.module_hparams)
        self.module_hparams.update(default_hparams)
        self.vect = None

    def fit(self, docs):
        self.vect = Pipeline([('count_vect', CountVectorizer(min_df=self.module_hparams['min_count'])),
                              ('tf_idf', TfidfTransformer())])
        self.vect.fit(docs)
        print('vocab_size: %s' % len(self.vect.steps[0][1].get_feature_names()))

    def transform(self, docs):
        return self.vect.transform(docs)

    def fit_and_transform(self, docs):
        return self.vect.fit_transform(docs)

    @staticmethod
    def load_from_hparams(hparams):
        t = BagWordIndexingTransform(hparams)
        t.module_hparams = hparams[constants.INDEXING_TRANSFORM]

        file_name = t.module_hparams['file_name']
        directory = hparams[constants.GLOBAL]['tmp_dir']
        with open(directory + '/' + file_name, 'rb') as i_f:
            t.vect = pickle.load(i_f)
        return t

    def save(self):
        if 'file_name' not in self.module_hparams:
            self.module_hparams['file_name'] = 'vect.pkl'
        with open(self.root_hparams[constants.GLOBAL]['tmp_dir'] + '/' + self.module_hparams['file_name'], 'wb') as o_f:
            pickle.dump(self.vect, o_f)


class WordEmbeddingIndexingTransform(IndexingTransform):
    def __init__(self, hparams):
        IndexingTransform.__init__(self, hparams)
        default_hparams = dict(size=300, window=10, min_count=5, workers=8, level='word')
        default_hparams.update(self.module_hparams)
        self.module_hparams.update(default_hparams)
        self.voc = None

    def transform(self, docs):
        return self.voc.docs2idx(docs)

    def fit(self, docs):
        print('Fitting...')
        print('Current hparams: %s' % self.module_hparams)
        print('There are %s docs' % len(docs))

        assert isinstance(docs, list)
        assert isinstance(docs[0], str)
        if self.module_hparams['level'] == 'word':
            all_docs = [doc.split() for doc in docs]
            model = Word2Vec(all_docs, size=self.module_hparams['size'], window=self.module_hparams['window'],
                             min_count=self.module_hparams['min_count'], workers=self.module_hparams['workers'])
            wv = model.wv
            del model

            self.voc = Voc(tokenize_func=Voc.WORD_LV_TOK_FUNC, space_char=Voc.WORD_LV_SPACE_CHR)
            padding = '__padding__'
            oov = '__oov__'

            self.module_hparams['padding_idx'] = 0
            self.module_hparams['oov_idx'] = 1
            self.voc.build_from_tokens([padding, oov] + wv.index2word, padding_idx=self.module_hparams['padding_idx'],
                                       oov_idx=self.module_hparams['oov_idx'])

            padding_vector = np.zeros(self.module_hparams['size'])
            trained_weights = wv.syn0
            oov_vector = np.random.normal(loc=trained_weights.mean(), scale=trained_weights.std(), size=(self.module_hparams['size'],))
            self.voc.add_embedding_weights(np.concatenate((padding_vector.reshape(1, -1), oov_vector.reshape(1, -1), trained_weights), axis=0))

            self.voc.freeze()
            self.module_hparams['vocab_size'] = len(self.voc._index2word)
            self.describe(docs, self.voc)

        elif self.module_hparams['level'] == 'char':
            raise NotImplemented('Only word is supported for now, while hparam.level=%s', self.module_hparams['level'])
        else:
            raise NotImplemented('Only word is supported for now, while hparam.level=%s', self.module_hparams['level'])

    def fit_and_transform(self, docs):
        self.fit(docs)
        return self.transform(docs)

    def save(self, path_to_output):
        self.voc.dump(path_to_output)

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
        return str(self.module_hparams)


if __name__ == '__main__':
    hparams = dict()
    hparams['pseudo_transform'] = dict(x=1, y=2)
    xx = WordEmbeddingIndexingTransform(hparams)
    print(hparams)

    hparams = dict()
    hparams['pseudo_transform'] = dict(x=1, y=2)
    xx = BagWordIndexingTransform(hparams)
    print(hparams)
