import json

import pandas as pd
from sklearn.metrics import classification_report

from naruto_skills.quick_training import constants


class Model:
    def __init__(self, hparams, clfs):
        """

        :param hparams:
        :param clfs: Should contain in order: split, preprocess, indexing, batching, magic
        """
        self.hparams = hparams
        self.pipeline = clfs

        self.split_comp = None
        self.preprocess_comp = None
        self.word_embed_indexing_comp = None
        self.batching_comp = None
        self.magic_comp = None

    def train(self):
        self.split_comp = self.pipeline[0](self.hparams)
        self.split_comp.process()
        df_train = pd.read_csv(self.hparams[constants.GLOBAL][constants.GLOBAL_DIRECTOR] + '/' +
                               self.hparams[constants.SPLIT_TRAIN_TEST_COMPONENT]['train_name'])

        docs, labels = list(df_train.iloc[:, 0]), list(df_train.iloc[:, 1])

        self.preprocess_comp = self.pipeline[1](self.hparams)
        docs = self.preprocess_comp.process(docs)

        self.word_embed_indexing_comp = self.pipeline[2](self.hparams)
        self.word_embed_indexing_comp.train(docs)
        docs = self.word_embed_indexing_comp.process(docs)
        self.batching_comp = self.pipeline[3](self.hparams)
        data_loader = self.batching_comp.process(docs, labels)
        self.magic_comp = self.pipeline[4](self.hparams)
        self.magic_comp.fit(data_loader)

        print('\nHparams: %s' % json.dumps(self.hparams, indent=4))

        self.batching_comp.component_hparams['shuffle'] = False

    def extract_predictor(self):
        return Predictor(self.preprocess_comp, self.word_embed_indexing_comp, self.batching_comp, self.magic_comp)


class Predictor:
    def __init__(self, *trained_components):
        self.trained_components = trained_components

    def predict(self, docs):
        data = docs
        for comp in self.trained_components:
            data = comp.process(data)
        return data
