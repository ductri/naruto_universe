import pandas as pd

from . import SplitTrainTestComponent
from .. import utils

print = utils.get_printer(__name__)


class FileMappingComponent(SplitTrainTestComponent):
    default_hparams = dict(train_name='train.csv', test_name='test.csv', eval_name='eval.csv')

    def __init__(self, hparams):
        SplitTrainTestComponent.__init__(self, hparams)
        self.is_train = True

    def process(self, message=None, *args, **kwargs):
        pass


class SimpleSplitComponent(SplitTrainTestComponent):
    default_hparams = dict(test=0.1, eval=0.1, train_name='train.csv', test_name='test.csv', eval_name='eval.csv', input_file='tmp.csv')

    def __init__(self, hparams):
        SplitTrainTestComponent.__init__(self, hparams)

    def process(self, message=None, **kwargs):
        print('Fitting...')

        df = pd.read_csv(self.component_hparams['input_file'])
        print('Total: %s' % df.shape[0])
        df = df.sample(df.shape[0])
        test_size = int(df.shape[0] * self.component_hparams['test'])
        eval_size = int(df.shape[0] * self.component_hparams['eval'])
        df_test = df.iloc[:test_size, :]
        df_eval = df.iloc[test_size:test_size+eval_size, :]
        df_train = df.iloc[test_size + eval_size:, :]

        df_test.to_csv(self.component_hparams['output_dir'] + '/%s' % self.component_hparams['test_name'], index=None)
        df_eval.to_csv(self.component_hparams['output_dir'] + '/%s' % self.component_hparams['eval_name'], index=None)
        df_train.to_csv(self.component_hparams['output_dir'] + '/%s' % self.component_hparams['train_name'], index=None)
        print('Train: %s \t Eval: %s \t Test: %s' % (df_train.shape[0], df_eval.shape[0], df_test.shape[0]))
        print('Finished ... Output at: %s' % self.component_hparams['output_dir'])
        return df_train, df_eval, df_test
