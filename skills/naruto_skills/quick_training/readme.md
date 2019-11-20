I feel what I have been coding in this module  unintentionally served the same purpose with Tensor2Tensor framework

Just give me dataset, I will instantly create a model for you!!!
Here are some assumptions and corresponding implication:
- Dataset is not huge: Not focusing on computational optimization

# Examples
```yaml
from collections import defaultdict
import json

from quick_training.task import SimpleTask, SimpleTask2, LSTMTask
from quick_training import constants


if __name__ == '__main__':

    hparams = defaultdict(lambda: {})
    hparams[constants.GLOBAL] = dict()
    hparams[constants.GLOBAL][constants.GLOBAL_DIRECTOR] = 'tmp/'
    hparams[constants.GLOBAL][constants.GLOBAL_HPARAMS] = 'hparams.txt'
    hparams[constants.INITIAL_TRANSFORM]['train_name'] = 'sample_train.csv'
    hparams[constants.INITIAL_TRANSFORM]['test_name'] = 'sample_test.csv'
    hparams[constants.INITIAL_TRANSFORM]['num_classes'] = 3
    hparams[constants.INDEXING_TRANSFORM]['min_count'] = 5
    hparams[constants.MAGICAL_TRANSFORM]['num_epochs'] = 5

    task = LSTMTask(hparams)
    task.train()
    # task.save()
    task.evaluate()
    hparams_file = hparams[constants.GLOBAL][constants.GLOBAL_DIRECTOR] + '/' \
                   + hparams[constants.GLOBAL][constants.GLOBAL_HPARAMS]
    with open(hparams_file, 'rt') as i_f:
        hparams1 = json.load(i_f)
```