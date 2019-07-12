import logging
import os
from datetime import datetime

import torch


class TrainingChecker:

    def __init__(self, model, root_dir, init_score):
        """
        Higher is better
        :param model:
        :param root_dir:
        :param init_score:
        """
        self._model = model
        self._score = init_score
        self._step = None

        self.__path_to_dir_save = root_dir
        logging.info('Model will be saved at %s' % self.__path_to_dir_save)
        os.makedirs(self.__path_to_dir_save)

    def update(self, score, step):
        """
        Higher is better
        :param score:
        :param step:
        :return:
        """
        if score > self._score:
            self._score = score
            self._step = step
            path_to_save = self.save_model()
            logging.info('New best score: %s', self._score)
            logging.info('Saved model at %s', path_to_save)
        else:
            logging.info('Current best score is %s at %s', self._score, self.__get_file_name())

    def best(self):
        return self._score, self._step

    def __get_file_name(self):
        return os.path.join(self.__path_to_dir_save, '%s.pt' % self._step)

    def save_model(self):
        file_name = self.__get_file_name()
        torch.save({
            'model_state_dict': self._model.state_dict(),
            'optimizer': self._model.optimizer.state_dict(),
            'step': self._step,
            'best_score': self._score
        }, file_name)
        return file_name
