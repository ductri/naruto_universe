from nltk.tokenize import word_tokenize

from . import utils, PreprocessComponent


print = utils.get_printer(__name__)


class SimplePreprocessComponent(PreprocessComponent):
    default_hparams = dict(max_length=100)

    def __init__(self, hparams):
        PreprocessComponent.__init__(self, hparams)
        self.is_train = True

    def process(self, all_docs, **kwargs):
        print('Fitting %s docs ...' % len(all_docs))

        all_docs = [word_tokenize(doc) for doc in all_docs]
        all_docs = [doc[:self.component_hparams['max_length']] for doc in all_docs]
        all_docs = [' '.join(doc).lower() for doc in all_docs]

        print('Finished ...')
        return all_docs

    def persist(self):
        pass
