import pandas as pd

from quick_training import constants
from quick_training.split_transform import SimpleSplitTransform
from quick_training.preprocess_transform import PreprocessTransform
from quick_training.indexing_transform import WordEmbeddingIndexingTransform, BagWordIndexingTransform
from quick_training.batching_transform import BatchingTransform
from quick_training.magical_transform import SimpleLogisticRegression


if __name__ == '__main__':
    hparams = dict()
    hparams[constants.GLOBAL] = dict()
    hparams[constants.GLOBAL][constants.GLOBAL_DIRECTOR] = 'tmp/'

    transformer_0 = SimpleSplitTransform(hparams)
    transformer_1 = PreprocessTransform(hparams)
    transformer_2 = BagWordIndexingTransform(hparams)
    # transformer_3 = BatchingTransform(hparams)
    transformer_4 = SimpleLogisticRegression(hparams)

    transformer_0.transform('tmp/sample.csv')

    df_train = pd.read_csv('tmp/train.csv')
    df_test = pd.read_csv('tmp/test.csv')

    docs, labels = list(df_train.iloc[:, 0]), list(df_train.iloc[:, 1])

    preprocessed_docs = transformer_1.transform(docs)

    transformer_2.fit(preprocessed_docs)
    indexing_docs = transformer_2.transform(preprocessed_docs)

    # batchs = transformer_3.transform(indexing_docs, labels)
    transformer_4.fit(indexing_docs, labels)

    print('Hparams: %s' % hparams)


