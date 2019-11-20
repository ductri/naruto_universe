from collections import defaultdict
import json

from quick_training.task import SimpleTask, SimpleTask2, LSTMTask
from quick_training import constants


if __name__ == '__main__':

    # hparams = defaultdict(lambda: {})
    # hparams[constants.GLOBAL] = dict()
    # hparams[constants.GLOBAL][constants.GLOBAL_DIRECTOR] = 'tmp/tmmp/'
    # hparams[constants.INITIAL_TRANSFORM]['input_file'] = 'tmp/sample.csv'
    #
    # task = SimpleTask(hparams)
    # task.train()
    # task.save()
    # task.evaluate()
    # file_name = hparams[constants.GLOBAL][constants.GLOBAL_DIRECTOR] + '/' + hparams[constants.GLOBAL][
    #     constants.GLOBAL_HPARAMS]
    # with open(file_name, 'rt') as i_f:
    #     hparams1 = json.load(i_f)
    # procedure1 = SimpleTask.load_from_hparams(hparams1)
    # procedure1.evaluate()
    #
    # hparams = defaultdict(lambda: {})
    # hparams[constants.GLOBAL] = dict()
    # hparams[constants.GLOBAL][constants.GLOBAL_DIRECTOR] = 'tmp/'
    # hparams[constants.INITIAL_TRANSFORM]['train_name'] = 'new_training_data.csv'
    # hparams[constants.INITIAL_TRANSFORM]['test_name'] = 'test_dataset_1.9.csv'
    # hparams[constants.INDEXING_TRANSFORM]['min_count'] = 2
    #
    # task = SimpleTask2(hparams)
    # task.train()
    # task.save()
    # task.evaluate()
    # file_name = hparams[constants.GLOBAL][constants.GLOBAL_DIRECTOR] + '/' + hparams[constants.GLOBAL][
    #     constants.GLOBAL_HPARAMS]
    # with open(file_name, 'rt') as i_f:
    #     hparams1 = json.load(i_f)
    #
    # procedure1 = SimpleTask2.load_from_hparams(hparams1)
    # procedure1.evaluate()

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
    # procedure1 = SimpleTask2.load_from_hparams(hparams1)
    # procedure1.evaluate()