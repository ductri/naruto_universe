import pandas as pd

from naruto_skills.quick_training import utils
from naruto_skills.quick_training import constants

print = utils.get_printer(__name__)


class InitialTransform:
    def __init__(self, hparams):
        self.root_hparams = hparams
        self.module_hparams = hparams[constants.INITIAL_TRANSFORM]

    def fit(self, *args):
        raise NotImplemented()

    def transform(self, *args):
        raise NotImplemented


class FileMappingTransform(InitialTransform):
    def __init__(self, hparams):
        InitialTransform.__init__(self, hparams)
        default_hparams = dict(train_name='train.csv', test_name='test.csv', eval_name='eval.csv',
                               output_dir=self.root_hparams[constants.GLOBAL][constants.GLOBAL_DIRECTOR])
        default_hparams.update(self.module_hparams)
        self.module_hparams.update(default_hparams)

    def transform(self):
        print('Already done')


class SimpleSplitTransform(InitialTransform):
    def __init__(self, hparams):
        InitialTransform.__init__(self, hparams)
        default_hparams = dict(test=0.1, eval=0.1, output_dir=self.root_hparams[constants.GLOBAL][constants.GLOBAL_DIRECTOR],
                                    train_name='train.csv', test_name='test.csv', eval_name='eval.csv', input_file='tmp.csv')
        default_hparams.update(self.module_hparams)
        self.module_hparams.update(default_hparams)

    def transform(self):
        print('Fitting...')

        df = pd.read_csv(self.module_hparams['input_file'])
        print('Total: %s' % df.shape[0])
        df = df.sample(df.shape[0])
        test_size = int(df.shape[0] * self.module_hparams['test'])
        eval_size = int(df.shape[0] * self.module_hparams['eval'])
        df_test = df.iloc[:test_size, :]
        df_eval = df.iloc[test_size:test_size+eval_size, :]
        df_train = df.iloc[test_size + eval_size:, :]

        df_test.to_csv(self.module_hparams['output_dir'] + '/%s' % self.module_hparams['test_name'], index=None)
        df_eval.to_csv(self.module_hparams['output_dir'] + '/%s' % self.module_hparams['eval_name'], index=None)
        df_train.to_csv(self.module_hparams['output_dir'] + '/%s' % self.module_hparams['train_name'], index=None)
        print('Train: %s \t Eval: %s \t Test: %s' % (df_train.shape[0], df_eval.shape[0], df_test.shape[0]))
        print('Finished ... Output at: %s' % self.module_hparams['output_dir'])
        return df_train, df_eval, df_test
