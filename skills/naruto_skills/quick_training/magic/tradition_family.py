import pickle

import numpy as np
from sklearn.linear_model import LogisticRegression

from . import MagicComponent, constants
from .. import utils


print = utils.get_printer(__name__)


class SimpleLogisticRegression(MagicComponent):
    default_hparams = dict(num_epochs=3, embedding_dim=300, hidden_size=300, num_layers=2, bidirectional=True,
                           dropout_rate=0.3, learning_rate=1e-3, device='cpu')

    def __init__(self, hparams):
        MagicComponent.__init__(self, hparams)
        self._clf = None
        self.__restore_if_possible()

    def process(self, message, *args, **kwargs):
        return self._clf.predict(message)

    def fit(self, training_data, *args, **kwargs):
        X, y = training_data
        y = np.array(y)
        self._clf = LogisticRegression()
        self._clf.fit(X, y)

    def persist(self):
        if constants.CONST_DUMPED_FILE not in self.component_hparams:
            self.component_hparams[constants.CONST_DUMPED_FILE] = 'logistics_regression.pkl'
        path_to_file = self.root_hparams[constants.GLOBAL]['tmp_dir'] + '/' + self.component_hparams[constants.CONST_DUMPED_FILE]
        with open(path_to_file, 'wb') as o_f:
            pickle.dump(self._clf, o_f)

    def __restore_if_possible(self):
        if constants.CONST_DUMPED_FILE in self.component_hparams:
            path_to_file = self.root_hparams[constants.GLOBAL]['tmp_dir'] + '/' + self.component_hparams[
                constants.CONST_DUMPED_FILE]
            with open(path_to_file, 'rb') as i_f:
                self._clf = pickle.load(i_f)
            print('Load trained model successfully')
