import pandas as pd

from quick_training import utils
from quick_training import constants

print = utils.get_printer(__name__)


class SplitTransform:
    def __init__(self, hparams):
        hparams[constants.SPLIT_TRANSFORM] = dict()
        self.root_hparams = hparams
        self.module_hparams = hparams[constants.SPLIT_TRANSFORM]

    def fit(self, *args):
        raise NotImplemented()

    def transform(self, *args):
        raise NotImplemented


class SimpleSplitTransform(SplitTransform):
    def __init__(self, hparams):
        SplitTransform.__init__(self, hparams)
        self.module_hparams['test'] = 0.1
        self.module_hparams['eval'] = 0.1
        self.module_hparams['output_dir'] = self.root_hparams[constants.GLOBAL][constants.GLOBAL_DIRECTOR]

    def transform(self, path_to_file):
        print('Fitting...')

        df = pd.read_csv(path_to_file)
        print('Total: %s' % df.shape[0])
        df = df.sample(df.shape[0])
        test_size = int(df.shape[0] * self.module_hparams['test'])
        eval_size = int(df.shape[0] * self.module_hparams['eval'])
        df_test = df.iloc[:test_size, :]
        df_eval = df.iloc[test_size:test_size+eval_size, :]
        df_train = df.iloc[test_size + eval_size:, :]

        df_test.to_csv(self.module_hparams['output_dir'] + '/test.csv', index=None)
        df_eval.to_csv(self.module_hparams['output_dir'] + '/eval.csv', index=None)
        df_train.to_csv(self.module_hparams['output_dir'] + '/train.csv', index=None)
        print('Train: %s \t Eval: %s \t Test: %s' % (df_train.shape[0], df_eval.shape[0], df_test.shape[0]))
        print('Finished ... Output at: %s' % self.module_hparams['output_dir'])
        return df_train, df_eval, df_test
