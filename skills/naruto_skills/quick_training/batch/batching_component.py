import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from . import constants, BatchingComponent
from .. import utils

print = utils.get_printer(__name__)


class SimpleBatchingComponent(BatchingComponent):
    default_hparams = dict(batch_size=64, shuffle=True)

    def __init__(self, hparams):
        BatchingComponent.__init__(self, hparams)

    def process(self, docs, *other_columns, **kwargs):
        print(f'Fitting {len(docs)} docs ...')
        dataset = PytorchDataset(docs, *other_columns)
        padding_idx = self.root_hparams[constants.INDEXING_COMPONENT]['padding_idx']

        def collate_fn(list_data):
            docs, *others = zip(*list_data)
            max_length = max([len(doc) for doc in docs])

            docs = [doc + [padding_idx] * (max_length - len(doc)) for doc in docs]
            data = [np.stack(col, axis=0) for col in (docs, *others)]
            data = [torch.from_numpy(col) for col in data]
            return data

        data_loader = DataLoader(dataset=dataset, collate_fn=collate_fn, batch_size=self.component_hparams['batch_size'],
                                 shuffle=self.component_hparams['shuffle'])
        print('Finished')
        print('There are %s batch/epoch' % len(data_loader))
        return data_loader


class AllPaddingBatchingComponent(BatchingComponent):
    default_hparams = dict(batch_size=64, shuffle=True, equal_length=50)

    def __init__(self, hparams):
        BatchingComponent.__init__(self, hparams)

    def process(self, docs, *other_columns, **kwargs):
        print(f'Fitting {len(docs)} docs ...')
        dataset = PytorchDataset(docs, *other_columns)
        padding_idx = self.root_hparams[constants.INDEXING_COMPONENT]['padding_idx']

        def collate_fn(list_data):
            docs, *others = zip(*list_data)
            max_length = self.component_hparams['equal_length']

            docs = [doc + [padding_idx] * (max_length - len(doc)) for doc in docs]
            docs = [doc[:self.component_hparams['equal_length']] for doc in docs]

            data = [np.stack(col, axis=0) for col in (docs, *others)]
            data = [torch.from_numpy(col) for col in data]
            return data

        data_loader = DataLoader(dataset=dataset, collate_fn=collate_fn, batch_size=self.component_hparams['batch_size'],
                                 shuffle=self.component_hparams['shuffle'])
        print('Finished')
        print('There are %s batch/epoch' % len(data_loader))
        return data_loader


class PytorchDataset(Dataset):
    def __init__(self, *columns):
        super(PytorchDataset, self).__init__()
        for i in range(len(columns)-1):
            assert len(columns[i]) == len(columns[i+1])
        self.__size = len(columns[0])
        self.rows = list(zip(*columns))

    def __len__(self):
        return self.__size

    def __getitem__(self, idx):
        return self.rows[idx]
