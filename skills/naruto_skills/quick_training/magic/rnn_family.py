import torch
from torch import nn, optim


from .pytorch_family import PytorchFamily
from .. import utils, constants

print = utils.get_printer(__name__)

CONST_DUMPED_FILE = 'dumped_file'


class SimpleLSTM(PytorchFamily):
    default_hparams = dict(num_epochs=3, embedding_dim=300, hidden_size=300, num_layers=2, bidirectional=True,
                           dropout_rate=0.3, learning_rate=1e-3, device='cpu')

    def __init__(self, hparams):
        PytorchFamily.__init__(self, hparams)
        self.output_prob = False

    def create_vital_elements(self):
        self.word_emb_layer = nn.Embedding(num_embeddings=self.root_hparams[constants.INDEXING_COMPONENT]['vocab_size'],
                                           embedding_dim=self.component_hparams['embedding_dim'])
        self.lstm_layer = nn.LSTM(input_size=self.component_hparams['embedding_dim'], hidden_size=self.component_hparams['hidden_size'],
                       num_layers=self.component_hparams['num_layers'], bidirectional=self.component_hparams['bidirectional'],
                       dropout=self.component_hparams['dropout_rate'])
        self.activate_fn = nn.ReLU()
        self.fc = nn.Linear(in_features=4*self.component_hparams['hidden_size'],
                            out_features=self.component_hparams['hidden_size'])
        self.output_mapping_layer = nn.Linear(in_features=self.component_hparams['hidden_size'],
                                              out_features=self.root_hparams[constants.GLOBAL][constants.GLOBAL_NUM_CLASSES])
        self.softmax_fn = nn.Softmax(dim=-1)
        self.dropout_fn = nn.Dropout(self.component_hparams['dropout_rate'])
        self.loss_fn = nn.CrossEntropyLoss(reduction='none')
        self.optimizer = optim.Adam(self.parameters(), lr=self.component_hparams['learning_rate'])

    def process(self, data_loader, *args, **kwargs):
        whole_preds = []
        for docs, *_ in data_loader:
            docs = docs.to(self.component_hparams['device'])
            logits = self._forward(docs)
            if not self.output_prob:
                output = logits.argmax(dim=-1).cpu().numpy()
            else:
                output = logits[:, 1].cpu().numpy()
            whole_preds.extend(output)

        return whole_preds

    def _forward(self, docs):
        tmp = self.word_emb_layer(docs)
        tmp = self.dropout_fn(tmp)

        # TODO check it if we do not permute ?
        tmp = tmp.permute(1, 0, 2)

        tmp, (h_n, c_n) = self.lstm_layer(tmp)
        batch_size = h_n.size(1)
        hidden_size = c_n.size(2)
        h_n = h_n.view(self.component_hparams['num_layers'], int(self.component_hparams['bidirectional'])+1,
                       batch_size, hidden_size)
        # c_n = c_n.view(self.component_hparams['num_layers'], int(self.component_hparams['bidirectional']) + 1,
        #                batch_size, hidden_size)
        h_n = h_n.permute(2, 0, 1, 3)
        tmp = h_n.reshape(batch_size, -1)
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
