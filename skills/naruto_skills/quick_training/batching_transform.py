import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from quick_training import constants, utils

print = utils.get_printer(__name__)


class BatchingTransform:
    def __init__(self, hparams):
        self.global_hparams = hparams
        self.global_hparams[constants.BATCHING_TRANSFORM] = dict(batch_size=64)
        self.hparams = hparams[constants.BATCHING_TRANSFORM]

    def transform(self, docs, labels):
        print(f'Fitting {len(docs)} docs ...')
        dataset = BatchingTransform.PytorchDataset(docs, labels)
        padding_idx = self.global_hparams[constants.INDEXING_TRANSFORM]['padding_idx']

        def collate_fn(list_data):
            docs, labels = zip(*list_data)
            max_length = max([len(doc) for doc in docs])

            docs = [doc + [padding_idx] * (max_length - len(doc)) for doc in docs]
            data = [np.stack(col, axis=0) for col in (docs, labels)]
            data = [torch.from_numpy(col) for col in data]
            return data

        data_loader = DataLoader(dataset=dataset, collate_fn=collate_fn, batch_size=self.hparams['batch_size'], shuffle=True)
        return data_loader

    class PytorchDataset(Dataset):
        def __init__(self, list_docs, list_labels):
            super(BatchingTransform.PytorchDataset, self).__init__()
            assert len(list_docs) == len(list_labels)
            self.docs = list_docs
            self.labels = list_labels

        def __len__(self):
            return len(self.docs)

        def __getitem__(self, idx):
            return self.docs[idx], self.labels[idx]
