import torch
from torch import nn


from . import MagicComponent
from .. import utils, constants

print = utils.get_printer(__name__)

CONST_DUMPED_FILE = 'dumped_file'


class PytorchFamily(MagicComponent, nn.Module):
    default_hparams = dict(num_epochs=3, embedding_dim=300,
                           dropout_rate=0.3, learning_rate=1e-3, device='cpu')

    def __init__(self, hparams):
        MagicComponent.__init__(self, hparams)
        nn.Module.__init__(self)
        self.__create_vital_elements()
        self.__restore_if_possible()

    def process(self, data_loader, *args, **kwargs):
        raise NotImplemented()

    def _forward(self, docs):
        raise NotImplemented()

    def _get_loss(self, docs, labels):
        raise NotImplemented()

    def _train_batch(self, docs, labels):
        """

        :param docs: shape == (batch_size, max_len)
        :param labels: shape == (batch_size, max_len)
        :return: loss.item()
        """
        raise NotImplemented()

    def fit(self, data_loader, *args, **kwargs):
        raise NotImplemented()

    def save(self):
        if CONST_DUMPED_FILE not in self.component_hparams:
            self.component_hparams[CONST_DUMPED_FILE] = 'model.pt'
        file_name = self.root_hparams[constants.GLOBAL][constants.GLOBAL_DIRECTOR] + self.component_hparams[CONST_DUMPED_FILE]
        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, file_name)

    def __restore_if_possible(self):
        if CONST_DUMPED_FILE in self.component_hparams:
            file_name = self.root_hparams[constants.GLOBAL][constants.GLOBAL_DIRECTOR] + self.component_hparams[
                CONST_DUMPED_FILE]
            self.load_state_dict(torch.load(file_name))
            nn.Module.eval(self)
            print('Load trained model successfully')

    def __create_vital_elements(self):
        raise NotImplemented()
