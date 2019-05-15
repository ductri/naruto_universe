import logging
import json
from pathlib import Path

import numpy as np
import pandas as pd

seed = 0

"""
These following features are not supported anymore, since it shoule be the responsibility of tensorflow tf.data: Shuffle, strictly equal size when getting batch

- There are no notion of label or X either. Each element in dataset is a tuple
- Aim to support iter to reduce the depending on memory size
"""


class NarutoDataset:
    PATH_TO_DATA = 'path_to_data'
    SIZE = 'size'
    NAME = 'name'

    def __init__(self, path_to_meta_data):
        """

        :param list_data: contain a list of tuple/np.narray/object
        :param name:
        """
        self._meta_data = json.load(open(path_to_meta_data, 'rt'))
        self.__rows = None

    def get_data_iterator(self, batch_size=64, num_epochs=1, is_strictly_equal=True):
        """

        :param X:
        :param y:
        :param batch_size: <0 for get all data in a batch
        :param num_epochs: <0 for an infinite generator
        :param is_strictly_equal: True mean that we will not return if len(data) < batch_size
        :return:
        """

        if num_epochs > 0:
            logging.info('Dataset "%s": will generate %s steps', self._meta_data[NarutoDataset.NAME],
                         self.estimate_number_of_steps(batch_size, num_epochs))
            for _ in range(num_epochs):
                return self.__generate_one_epoch(batch_size)
        else:
            logging.info('We are going to generate infinite data.')
            while True:
                return self.__generate_one_epoch(batch_size)

    def __generate_one_epoch(self, batch_size):
        list_chunks = pd.read_csv(self._meta_data[NarutoDataset.PATH_TO_DATA], lineterminator='\n', sep='\t',
                                  header=None,
                                  chunksize=batch_size)
        for df_chunk in list_chunks:
            yield list([json.loads(row) for row in df_chunk[0]])

    def estimate_number_of_steps(self, batch_size, num_epochs):
        if num_epochs < 0:
            return -1
        else:
            return np.ceil(self._meta_data[NarutoDataset.SIZE] / batch_size) * num_epochs

    def show_info(self, is_logging=True):
        display_func = logging.info if is_logging else print

        display_func(self._meta_data)

    @staticmethod
    def dump_from_generator(generator, data_dir, name, chunk_size=10000):
        """

        :param generator:
        :param data_dir:
        :param name:
        :param chunk_size: It's dependent on how much memory you want to save
        :return:
        """
        logging.info('Creating dataset named %s', name)
        dir_path = Path(data_dir)
        data_path = dir_path / ('%s_data.csv' % name)
        meta_data_path = dir_path / ('%s_meta_data.json' % name)

        data_f = data_path.open('wt', encoding='utf-8', newline='')
        counter = 0
        chunk = []
        for e in generator:
            counter += 1
            # It might be more general by let it defines how to serializing itself.
            # For simplicity, I just use json.dumps in this case
            chunk.append(json.dumps(e))

            if counter % chunk_size == 0:
                NarutoDataset.__chunk_to_disk(chunk, data_f)
                chunk = []
                logging.info('Wrote %s rows to disk', counter)

        if len(chunk) > 0:
            NarutoDataset.__chunk_to_disk(chunk, data_f)
            logging.info('Wrote %s rows to disk', counter)

        data_f.close()
        meta_data = {
            NarutoDataset.NAME: name,
            NarutoDataset.SIZE: counter,
            NarutoDataset.PATH_TO_DATA: str(data_path.absolute())}
        json.dump(meta_data, meta_data_path.open('wt'))
        logging.info('Created dataset named %s - Metadata is saved at %s', name, meta_data_path)

    @staticmethod
    def __chunk_to_disk(chunk, f_o):
        """

        :param chunk: a list
        :param f_o:
        :return:
        """
        pd.DataFrame({0: chunk}).to_csv(f_o, index=None, header=None, sep='\t')


def split_data_and_dump(data_generator, dir_data, eval_size=5000, test_size=5000, chunk_size=10000):
    """
    The order of implementation does matter. Don't reorder it without reasons
    :param data_generator:
    :param dir_data:
    :param eval_size:
    :param test_size:
    :param chunk_size:
    :return:
    """

    test_generator = (next(data_generator) for _ in range(test_size))
    NarutoDataset.dump_from_generator(test_generator, data_dir=dir_data, name='test', chunk_size=chunk_size)

    eval_generator = (next(data_generator) for _ in range(eval_size))
    NarutoDataset.dump_from_generator(eval_generator, data_dir=dir_data, name='eval', chunk_size=chunk_size)

    NarutoDataset.dump_from_generator(data_generator, data_dir=dir_data, name='train', chunk_size=chunk_size)


def split_data_and_dump_csv(path_to_csv, path_to_dir_output, eval_size, test_size, chunk_size=10000, preprocess_f=None):
    if preprocess_f is None:
        preprocess_f = lambda x: x

    def my_generator():
        chunks = pd.read_csv(path_to_csv, chunksize=chunk_size, header=None)
        for df in chunks:
            for row in df[0]:
                yield preprocess_f(row)

    data_generator = my_generator()
    split_data_and_dump(data_generator,
                        dir_data=path_to_dir_output,
                        eval_size=eval_size,
                        test_size=test_size, chunk_size=chunk_size)
