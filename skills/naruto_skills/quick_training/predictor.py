import pandas as pd

from quick_training import constants
from quick_training.split_transform import SimpleSplitTransform
from quick_training.preprocess_transform import PreprocessTransform
from quick_training.indexing_transform import WordEmbeddingIndexingTransform, BagWordIndexingTransform
from quick_training.batching_transform import BatchingTransform
from quick_training.magical_transform import SimpleLogisticRegression


class Predictor:
    def __init__(self, hparams):
        self.transformer_1 = PreprocessTransform.load_from_hparams(hparams)
        self.transformer_2 = BagWordIndexingTransform.load_from_hparams(hparams)
        self.transformer_4 = SimpleLogisticRegression.load_from_hparams(hparams)

    def predict(self, docs):
        docs = self.transformer_1.transform(docs)
        docs = self.transformer_2.transform(docs)
        docs = self.transformer_4.transform(docs)
        return docs
