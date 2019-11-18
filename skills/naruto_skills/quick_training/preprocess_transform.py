from nltk.tokenize import word_tokenize

from quick_training import utils
from quick_training import constants

print = utils.get_printer(__name__)


class PreprocessTransform:
    def __init__(self, hparams):
        self.global_hparams = hparams
        self.module_hparams = self.global_hparams[constants.PREPROCESS_TRANSFORM]

        self.module_hparams['max_length'] =100

    def transform(self, all_docs):
        print('Fitting %s docs ...' % len(all_docs))

        all_docs = [word_tokenize(doc) for doc in all_docs]
        all_docs = [doc[:self.module_hparams['max_length']] for doc in all_docs]
        all_docs = [' '.join(doc).lower() for doc in all_docs]

        print('Finished ...')
        return all_docs

    @staticmethod
    def load_from_hparams(hparams):
        t = PreprocessTransform(hparams)
        t.module_hparams = hparams[constants.PREPROCESS_TRANSFORM]
        return t
