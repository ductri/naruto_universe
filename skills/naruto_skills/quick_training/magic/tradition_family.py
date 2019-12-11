import numpy as np
from sklearn.linear_model import LogisticRegression

from . import MagicComponent


class SimpleLogisticRegression(MagicComponent):
    default_hparams = dict(num_epochs=3, embedding_dim=300, hidden_size=300, num_layers=2, bidirectional=True,
                           dropout_rate=0.3, learning_rate=1e-3, device='cpu')

    def __init__(self, hparams):
        MagicComponent.__init__(self, hparams)
        self._clf = None

    def process(self, message, *args, **kwargs):
        return self._clf.predict(message)

    def fit(self, training_data, *args, **kwargs):
        X, y = training_data
        y = np.array(y)
        self._clf = LogisticRegression()
        self._clf.fit(X, y)
