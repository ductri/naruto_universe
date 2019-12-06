from .. import constants
from ..component import Component


class SplitTrainTestComponent(Component):
    name = constants.SPLIT_TRAIN_TEST_COMPONENT

    def __init__(self, hparams):
        Component.__init__(self, hparams)
        if 'output_dir' not in self.component_hparams:
            self.component_hparams['output_dir'] = self.root_hparams[constants.GLOBAL]['output_dir']
