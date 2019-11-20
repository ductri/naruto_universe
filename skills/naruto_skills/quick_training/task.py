import pandas as pd
from sklearn.metrics import classification_report
import json

from quick_training import constants
from quick_training.initial_transform import SimpleSplitTransform, FileMappingTransform
from quick_training.preprocess_transform import SimplePreprocessTransform
from quick_training.indexing_transform import WordEmbeddingIndexingTransform, BagWordIndexingTransform
from quick_training.batching_transform import SimpleBatchingTransform
from quick_training.magical_transform.traditional_transform import SimpleLogisticRegression
from quick_training.magical_transform.deep_transform import SimpleLSTM
from quick_training import utils


print = utils.get_printer(__name__)


class TaskBase:
    def __init__(self, hparams):
        pass

    def train(self, *args):
        raise NotImplemented()

    def report(self, *args):
        raise NotImplemented()


class SimpleTask(TaskBase):
    def __init__(self, hparams):
        TaskBase.__init__(self, hparams)

        self.transformer_0 = SimpleSplitTransform(hparams)
        self.transformer_1 = SimplePreprocessTransform(hparams)
        self.transformer_2 = BagWordIndexingTransform(hparams)
        self.transformer_3 = SimpleLogisticRegression(hparams)
        self.hparams = hparams

    def train(self):
        self.transformer_0.transform()
        df_train = pd.read_csv(self.hparams[constants.GLOBAL][constants.GLOBAL_DIRECTOR] +
                               self.hparams[constants.INITIAL_TRANSFORM]['train_name'])

        docs, labels = list(df_train.iloc[:, 0]), list(df_train.iloc[:, 1])
        docs = self.transformer_1.transform(docs)

        self.transformer_2.fit(docs)
        docs = self.transformer_2.transform(docs)

        self.transformer_3.fit(docs, labels)

        print('\nHparams: %s' % json.dumps(self.hparams, indent=4))

    def predict(self, docs):
        docs = self.transformer_1.transform(docs)
        docs = self.transformer_2.transform(docs)
        docs = self.transformer_3.transform(docs)
        return docs

    def evaluate(self):
        df_test = pd.read_csv(self.hparams[constants.GLOBAL][constants.GLOBAL_DIRECTOR] +
                              self.hparams[constants.INITIAL_TRANSFORM]['test_name'])
        df_test.dropna(inplace=True)
        docs, labels = list(df_test.iloc[:, 0]), list(df_test.iloc[:, 1])
        pred = self.predict(docs)
        print('\n' + classification_report(y_true=labels, y_pred=pred))

    @staticmethod
    def load_from_hparams(hparams):
        model = SimpleTask(hparams)
        model.transformer_1 = SimplePreprocessTransform.load_from_hparams(hparams)
        model.transformer_2 = BagWordIndexingTransform.load_from_hparams(hparams)
        model.transformer_3 = SimpleLogisticRegression.load_from_hparams(hparams)
        return model

    def save(self):
        self.transformer_2.save()
        self.transformer_3.save()
        if constants.GLOBAL_HPARAMS not in self.hparams[constants.GLOBAL]:
            self.hparams[constants.GLOBAL][constants.GLOBAL_HPARAMS] = 'hparams.txt'
        file_name = self.hparams[constants.GLOBAL][constants.GLOBAL_HPARAMS]
        directory = self.hparams[constants.GLOBAL][constants.GLOBAL_DIRECTOR]
        with open(directory + '/' + file_name, 'wt') as o_f:
            json.dump(self.hparams, o_f)


class SimpleTask2(SimpleTask):
    def __init__(self, hparams):
        SimpleTask.__init__(self, hparams)
        self.transformer_0 = FileMappingTransform(hparams)


class LSTMTask(TaskBase):
    def __init__(self, hparams):
        TaskBase.__init__(self, hparams)

        self.transformer_0 = None
        self.transformer_1 = None
        self.transformer_2 = None
        self.transformer_3 = None
        self.transformer_33 = None
        self.transformer_4 = None

        self.hparams = hparams

    def train(self):
        self.transformer_0 = FileMappingTransform(self.hparams)
        self.transformer_0.transform()
        df_train = pd.read_csv(self.hparams[constants.GLOBAL][constants.GLOBAL_DIRECTOR] +
                               self.hparams[constants.INITIAL_TRANSFORM]['train_name'])

        docs, labels = list(df_train.iloc[:, 0]), list(df_train.iloc[:, 1])

        self.transformer_1 = SimplePreprocessTransform(self.hparams)
        docs = self.transformer_1.transform(docs)

        self.transformer_2 = WordEmbeddingIndexingTransform(self.hparams)
        self.transformer_2.fit(docs)
        docs = self.transformer_2.transform(docs)

        self.transformer_3 = SimpleBatchingTransform(self.hparams)
        data_loader = self.transformer_3.transform(docs, labels)

        self.transformer_4 = SimpleLSTM(self.hparams)
        self.transformer_4.fit(data_loader)

        print('\nHparams: %s' % json.dumps(self.hparams, indent=4))

        self.transformer_3.module_hparams['shuffle'] = False
        print('\nPerformance on train: %s' % classification_report(y_true=list(df_train.iloc[:, 1]),
                                             y_pred=self.predict(list(df_train.iloc[:, 0]))))

    def predict(self, docs):
        docs = self.transformer_1.transform(docs)
        docs = self.transformer_2.transform(docs)
        docs = self.transformer_3.transform(docs)
        docs = self.transformer_4.transform(docs)
        return docs

    def evaluate(self):
        df_test = pd.read_csv(self.hparams[constants.GLOBAL][constants.GLOBAL_DIRECTOR] +
                              self.hparams[constants.INITIAL_TRANSFORM]['test_name'])
        df_test.dropna(inplace=True)
        docs, labels = list(df_test.iloc[:, 0]), list(df_test.iloc[:, 1])
        pred = self.predict(docs)
        print('\n' + classification_report(y_true=labels, y_pred=pred))

    @staticmethod
    def load_from_hparams(hparams):
        model = SimpleTask(hparams)
        model.transformer_1 = SimplePreprocessTransform.load_from_hparams(hparams)
        model.transformer_2 = BagWordIndexingTransform.load_from_hparams(hparams)
        model.transformer_3 = SimpleLogisticRegression.load_from_hparams(hparams)
        return model

    def save(self):
        self.transformer_2.save()
        self.transformer_3.save()
        if constants.GLOBAL_HPARAMS not in self.hparams[constants.GLOBAL]:
            self.hparams[constants.GLOBAL][constants.GLOBAL_HPARAMS] = 'hparams.txt'
        file_name = self.hparams[constants.GLOBAL][constants.GLOBAL_HPARAMS]
        directory = self.hparams[constants.GLOBAL][constants.GLOBAL_DIRECTOR]
        with open(directory + '/' + file_name, 'wt') as o_f:
            json.dump(self.hparams, o_f)
