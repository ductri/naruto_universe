from sklearn.linear_model.logistic import LogisticRegression
from sklearn.metrics import classification_report
import pickle

from quick_training import constants
from quick_training import utils


print = utils.get_printer(__name__)


class MagicalTransform:
    def __init__(self, hparams):
        self.root_hparams = hparams
        self.module_hparams = hparams[constants.MAGICAL_TRANSFORM]

    def fit(self, *args):
        raise NotImplemented()

    def transform(self, docs):
        raise NotImplemented

    def self_report(self):
        pass

    def eval_report(self, *args):
        raise NotImplemented


class SimpleLogisticRegression(MagicalTransform):
    def __init__(self, hparams):
        MagicalTransform.__init__(self, hparams)
        self.module_hparams['num_epochs'] = 10
        self.model = None

    def fit(self, docs, labels):
        self.model = LogisticRegression(max_iter=self.module_hparams['num_epochs'])
        self.model.fit(docs, labels)
        print('\n' + classification_report(y_true=labels, y_pred=self.model.predict(docs)))

    def transform(self, docs):
        return self.model.predict(docs)

    def save(self):
        if 'file_name' not in self.module_hparams:
            self.module_hparams['file_name'] = 'model.pkl'
        file_name = self.module_hparams['file_name']
        directory = self.root_hparams[constants.GLOBAL][constants.GLOBAL_DIRECTOR]
        with open(directory + '/' + file_name, 'wb') as o_f:
            pickle.dump(self.model, o_f)

    @staticmethod
    def load_from_hparams(hparams):
        t = SimpleLogisticRegression(hparams)
        t.module_hparams = hparams[constants.MAGICAL_TRANSFORM]

        file_name = t.module_hparams['file_name']
        directory = t.root_hparams[constants.GLOBAL][constants.GLOBAL_DIRECTOR]
        with open(directory + '/' + file_name, 'rb') as i_f:
            t.model = pickle.load(i_f)
        return t
