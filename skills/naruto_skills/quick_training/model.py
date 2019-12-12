import json

import pandas as pd

from naruto_skills.quick_training import constants
from naruto_skills.quick_training.preprocess.preprocess_component import *
from naruto_skills.quick_training.indexing.indexing_component import *
from naruto_skills.quick_training.batch.batching_component import *
from naruto_skills.quick_training.magic.tradition_family import *
from naruto_skills.quick_training.magic.rnn_family import *
from naruto_skills.quick_training.magic.cnn_family import *


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
                               self.hparams[constants.SPLIT_TRAIN_TEST_COMPONENT]['train_name'], lineterminator='\n')

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
        for comp in self.trained_components:
            comp.component_hparams['class_name'] = comp.__class__.__name__

    def predict(self, docs):
        data = docs
        for comp in self.trained_components:
            data = comp.process(data)
        return data

    def persist(self):
        for comp in self.trained_components:
            comp.persist()

    @staticmethod
    def load(hparams):
        components_for_predict = [constants.PREPROCESS_COMPONENT, constants.INDEXING_COMPONENT, constants.BATCHING_COMPONENT, constants.MAGIC_COMPONENT]
        components = [eval(hparams[comp_name]['class_name'])(hparams) for comp_name in components_for_predict]
        return Predictor(*components)
