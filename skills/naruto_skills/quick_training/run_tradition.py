from collections import defaultdict

from naruto_skills.quick_training import constants
from naruto_skills.quick_training.model import Model, Predictor
from naruto_skills.quick_training.split_train_test.split_train_test import FileMappingComponent
from naruto_skills.quick_training.preprocess.preprocess_component import SimplePreprocessComponent
from naruto_skills.quick_training.indexing.indexing_component import BagWordIndexingComponent
from naruto_skills.quick_training.batch.batching_component import NonBatchingComponent
from naruto_skills.quick_training.magic.tradition_family import SimpleLogisticRegression


if __name__ == '__main__':

    hparams = defaultdict(lambda: {})
    hparams[constants.GLOBAL] = dict()

    # All configs below are compulsory
    hparams[constants.GLOBAL][constants.GLOBAL_DIRECTOR] = 'tmp/'
    hparams[constants.GLOBAL][constants.GLOBAL_HPARAMS] = 'hparams.txt'
    hparams[constants.GLOBAL][constants.GLOBAL_OUTPUT] = 'tmp/out/'
    hparams[constants.GLOBAL][constants.GLOBAL_NUM_CLASSES] = 3
    hparams[constants.GLOBAL][constants.GLOBAL_SEQUENCE_LENGTH] = 50

    # All configs below are optional
    hparams[constants.SPLIT_TRAIN_TEST_COMPONENT]['train_name'] = 'sample_train.csv'
    hparams[constants.SPLIT_TRAIN_TEST_COMPONENT]['test_name'] = 'sample_test.csv'
    hparams[constants.PREPROCESS_COMPONENT]['max_length'] = 50
    hparams[constants.INDEXING_COMPONENT]['min_count'] = 5
    hparams[constants.MAGIC_COMPONENT]['num_epochs'] = 2
    hparams[constants.MAGIC_COMPONENT]['fc__in_features'] = 2700

    model = Model(hparams, [FileMappingComponent, SimplePreprocessComponent, BagWordIndexingComponent,
                            NonBatchingComponent, SimpleLogisticRegression])
    model.train()
    predictor = model.extract_predictor()
    predictor.persist()
    del predictor

    predictor = Predictor.load(hparams)
    import pandas as pd
    from sklearn.metrics import classification_report
    df_test = pd.read_csv('tmp/sample_test.csv')
    y_pred = predictor.predict(list(df_test['content']))
    y_true = list(df_test['sentiment'])
    print(classification_report(y_true=y_true, y_pred=y_pred))
