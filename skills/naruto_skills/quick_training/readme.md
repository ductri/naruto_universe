I feel what I have been coding in this module  unintentionally served the same purpose with Tensor2Tensor framework

Just give me dataset, I will instantly create a model for you!!!
Here are some assumptions and corresponding implication:
- Dataset is not huge: Not focusing on computational optimization
- Follow [Rara](https://blog.rasa.com/enhancing-rasa-nlu-with-custom-components/) structure's philosophy

# Examples
```yaml
from collections import defaultdict

from naruto_skills.quick_training import constants
from naruto_skills.quick_training.model import Model, Predictor

from naruto_skills.quick_training.split_train_test.split_train_test import FileMappingComponent
from naruto_skills.quick_training.preprocess.preprocess_component import SimplePreprocessComponent
from naruto_skills.quick_training.indexing.indexing_component import WordEmbeddingIndexingComponent
from naruto_skills.quick_training.batch.batching_component import SimpleBatchingComponent
from naruto_skills.quick_training.magic.rnn_family import SimpleLSTM


if __name__ == '__main__':

    hparams = defaultdict(lambda: {})
    hparams[constants.GLOBAL] = dict()

    # All configs below are compulsory
    hparams[constants.GLOBAL][constants.GLOBAL_DIRECTOR] = 'tmp/'
    hparams[constants.GLOBAL][constants.GLOBAL_HPARAMS] = 'hparams.txt'
    hparams[constants.GLOBAL][constants.GLOBAL_OUTPUT] = 'tmp/out/'
    hparams[constants.GLOBAL][constants.GLOBAL_NUM_CLASSES] = 3

    # All configs below are optional
    hparams[constants.SPLIT_TRAIN_TEST_COMPONENT]['train_name'] = 'sample_train.csv'
    hparams[constants.SPLIT_TRAIN_TEST_COMPONENT]['test_name'] = 'sample_test.csv'
    hparams[constants.INDEXING_COMPONENT]['min_count'] = 5
    hparams[constants.MAGIC_COMPONENT]['num_epochs'] = 1

    model = Model(hparams, [FileMappingComponent, SimplePreprocessComponent, WordEmbeddingIndexingComponent,
                            SimpleBatchingComponent, SimpleLSTM])
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

```
