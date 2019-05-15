import logging
from sklearn.model_selection import train_test_split
import os

import numpy as np
import pandas as pd

seed = 0


class DataWithLabel:
    def __init__(self, X, y, name=''):
        assert len(X) == len(y)
        self.X = DataWithLabel.__convert_to_array(X)
        self.y = DataWithLabel.__convert_to_array(y)
        self.name = name

    @staticmethod
    def __convert_to_array(data):
        if not isinstance(data, np.ndarray):
            return np.array(data)
        return data

    def get_data_iterator(self, batch_size=64, num_epochs=1, is_strictly_equal=True):
        """

        :param X:
        :param y:
        :param batch_size: <0 for get all data in a batch
        :param num_epochs: <0 for an infinite generator
        :param is_strictly_equal: True mean that we will not return if len(data) < batch_size
        :return:
        """
        if is_strictly_equal:
            if batch_size < 0:
                batch_size = len(self.y)

            if self.X.shape[0] - batch_size < 0:
                raise Exception('Data len is only %s, while you are requiring %s for a batch' % (self.X.shape[0], batch_size))

        if num_epochs > 0:
            logging.info('Dataset "%s": will generate %s steps', self.name, self.estimate_number_of_steps(batch_size, num_epochs, is_strictly_equal))
            for _ in range(num_epochs):
                self.__shuffle_data()
                for batch in range(0, self.X.shape[0], batch_size):
                    if not is_strictly_equal:
                        yield self.X[batch: batch + batch_size], self.y[batch: batch + batch_size]
                    elif batch + batch_size <= self.X.shape[0]:
                        yield self.X[batch: batch+batch_size], self.y[batch: batch+batch_size]
        else:
            logging.info('We are going to generate infinite data. Shuffling happens every %s steps', int(self.X.shape[0]/batch_size))
            while True:
                self.__shuffle_data()
                for batch in range(0, self.X.shape[0], batch_size):
                    if batch + batch_size <= self.X.shape[0]:
                        yield self.X[batch: batch + batch_size], self.y[batch: batch + batch_size]

    def estimate_number_of_steps(self, batch_size, num_epochs, is_strictly_equal=True):
        if batch_size < 0:
            batch_size = len(self.y)
        if is_strictly_equal:
            if self.X.shape[0] - batch_size < 0:
                raise Exception('Data len is only %s, while you are requiring %s for a batch' % (self.X.shape[0], batch_size))
            return int(len(self.X) / batch_size) * num_epochs
        else:
            return np.ceil(len(self.X) / batch_size) * num_epochs

    def __shuffle_data(self):
        shuffled_index = list(range(self.X.shape[0]))
        np.random.shuffle(shuffled_index)
        self.X = self.X[shuffled_index]
        self.y = self.y[shuffled_index]

    def show_info(self, is_logging=True):
        display_func = logging.info if is_logging else print
        display_func('Data %s has size: %s' % (self.name, self.X.shape[0]))
        if self.y.shape[0] > 0:
            if isinstance(self.y[0], np.ndarray):
                display_func('Number of classes: %s' % (self.y.shape[1]))
            if np.isscalar(self.y[0]):
                display_func('\n%s', pd.DataFrame({'y': self.y})['y'].value_counts())

    def dump(self, path_to_dir, prefix=''):
        if not os.path.isdir(path_to_dir):
            os.makedirs(path_to_dir)
        path_to_file = os.path.join(path_to_dir, '{}_{}.npz'.format(prefix, self.name))
        np.savez(path_to_file, X=self.X, y=self.y)
        logging.info('Dumped dataset "%s" to %s', self.name, path_to_file)

    @staticmethod
    def create_from_npz(path_to_file, name=''):
        data = np.load(path_to_file)
        return DataWithLabel(X=data['X'], y=data['y'], name=name)


class Dataset:
    def __init__(self, data_with_label_train, data_with_label_eval, data_with_label_test, name=''):
        self.data_train = data_with_label_train
        self.data_eval = data_with_label_eval
        self.data_test = data_with_label_test
        self.name = name

    @staticmethod
    def create_from_entire_data(X, y, eval_prop=0.1, test_prop=0.1, name=''):
        """
        :param X:
        :param y:
        :param train_prop:
        :param eval_prop:
        :param test_prop:
        :return:
        """
        if eval_prop > 1.:
            eval_prop = int(eval_prop)
        if test_prop > 1.:
            test_prop = int(test_prop)

        X = np.array(X)
        y = np.array(y)

        X_train, X_remaning, y_train, y_remaning = train_test_split(X, y, test_size=eval_prop+test_prop, random_state=42)
        X_eval, X_test, y_eval, y_test = train_test_split(X_remaning, y_remaning, test_size=test_prop)
        data_with_label_train = DataWithLabel(X_train, y_train, 'train')
        data_with_label_eval = DataWithLabel(X_eval, y_eval, 'eval')
        data_with_label_test = DataWithLabel(X_test, y_test, 'test')

        return Dataset(data_with_label_train, data_with_label_eval, data_with_label_test, name)

    @staticmethod
    def create_from_npz(path_to_dir, prefix=''):
        data_with_label_train = DataWithLabel.create_from_npz(os.path.join(path_to_dir, '{}_train.npz'.format(prefix)), 'train')
        data_with_label_eval = DataWithLabel.create_from_npz(os.path.join(path_to_dir, '{}_eval.npz'.format(prefix)), 'eval')
        data_with_label_test = DataWithLabel.create_from_npz(os.path.join(path_to_dir, '{}_test.npz'.format(prefix)), 'test')
        return Dataset(data_with_label_train, data_with_label_eval, data_with_label_test)

    def show_info(self):
        self.data_train.show_info()
        self.data_eval.show_info()
        self.data_test.show_info()

    def dump(self, path_to_dir):
        self.data_train.dump(path_to_dir)
        self.data_eval.dump(path_to_dir)
        self.data_test.dump(path_to_dir)
