import torch
from torch import nn, optim

from . import constants
from .pytorch_family import PytorchFamily
from .. import utils

print = utils.get_printer(__name__)

CONST_DUMPED_FILE = 'dumped_file'


class SimpleCNN(PytorchFamily):
    default_hparams = dict(num_epochs=3, learning_rate=1e-3, device='cpu',
                           embedding_dim=300, out_channels=300, hidden_size=100, kernel_size=5, dropout_rate=0.3,
                           fc__in_features=6300)

    def __init__(self, hparams):
        PytorchFamily.__init__(self, hparams)

    def create_vital_elements(self):
        if not 'vocab_size' in self.component_hparams:
            self.component_hparams['vocab_size'] = self.root_hparams[constants.INDEXING_COMPONENT]['vocab_size']
        if not 'num_classes' in self.component_hparams:
            self.component_hparams['num_classes'] = self.root_hparams[constants.GLOBAL][constants.GLOBAL_NUM_CLASSES]
        if not 'sequence_length' in self.component_hparams:
            self.component_hparams['sequence_length'] = self.root_hparams[constants.GLOBAL][constants.GLOBAL_SEQUENCE_LENGTH]

        self.word_emb_layer = nn.Embedding(num_embeddings=self.component_hparams['vocab_size'],
                                           embedding_dim=self.component_hparams['embedding_dim'])
        self.cnn1 = nn.Conv1d(in_channels=self.component_hparams['embedding_dim'],
                             out_channels=self.component_hparams['out_channels'],
                             kernel_size=self.component_hparams['kernel_size'])

        self.cnn2 = nn.Conv1d(in_channels=self.component_hparams['embedding_dim'],
                             out_channels=self.component_hparams['out_channels'],
                             kernel_size=self.component_hparams['kernel_size'], stride=2)

        self.cnn3 = nn.Conv1d(in_channels=self.component_hparams['embedding_dim'],
                             out_channels=self.component_hparams['out_channels'],
                             kernel_size=self.component_hparams['kernel_size'], stride=2)

        self.activate_fn = nn.ReLU()
        self.fc = nn.Linear(in_features=self.component_hparams['fc__in_features'],
                            out_features=self.component_hparams['hidden_size'])
        self.output_mapping_layer = nn.Linear(in_features=self.component_hparams['hidden_size'],
                                              out_features=self.component_hparams['num_classes'])
        self.softmax_fn = nn.Softmax(dim=-1)
        self.dropout_fn = nn.Dropout(self.component_hparams['dropout_rate'])
        self.loss_fn = nn.CrossEntropyLoss(reduction='none')
        self.optimizer = optim.Adam(self.parameters(), lr=self.component_hparams['learning_rate'])

        self.root_hparams[constants.TRAINER_COMPONENT] = dict(num_epochs=self.component_hparams['num_epochs'],
                                                    learning_rate=self.component_hparams['learning_rate'],
                                                    device=self.component_hparams['device'])

    def process(self, data_loader, *args, **kwargs):
        whole_preds = []
        for docs, *_ in data_loader:
            docs = docs.to(self.component_hparams['device'])
            logits = self._forward(docs)
            output = logits.argmax(dim=-1).cpu().numpy()
            whole_preds.extend(output)

        return whole_preds

    def _forward(self, docs):
        tmp = self.word_emb_layer(docs)

        # (batch, length, emb)
        tmp = self.dropout_fn(tmp)

        # (batch, emb, length)
        tmp = tmp.permute(0, 2, 1)

        tmp = self.cnn1(tmp)
        tmp = self.cnn2(tmp)
        tmp = self.cnn3(tmp)

        batch_size = docs.size(0)
        tmp = tmp.reshape(batch_size, -1)
        tmp = self.fc(tmp)
        tmp = self.activate_fn(tmp)
        tmp = self.output_mapping_layer(tmp)
        return tmp

    def _get_loss(self, docs, labels):
        logits = self._forward(docs)
        loss = self.loss_fn(logits, labels)
        loss = loss.mean(dim=0)
        return loss

    def _train_batch(self, docs, labels):
        """

        :param docs: shape == (batch_size, max_len)
        :param labels: shape == (batch_size, max_len)
        :return:
        """
        nn.Module.train(self)
        self.optimizer.zero_grad()
        loss = self._get_loss(docs, labels)
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def fit(self, data_loader, *args, **kwargs):
        nn.Module.train(self, True)
        device = torch.device(self.component_hparams['device'])
        self.to(device)
        for epoch_idx in range(self.component_hparams['num_epochs']):
            print('Training epoch %s ...' % epoch_idx)
            l = 0
            for docs, labels in data_loader:
                docs = docs.to(device)
                labels = labels.to(device)
                l = self._train_batch(docs, labels)
            print('Loss: %.2f' % l)
        nn.Module.eval(self)
        print('Finished')