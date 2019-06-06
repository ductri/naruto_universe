"""
Based on TensorboardX for tensorboard mode (https://github.com/lanpa/tensorboardX/blob/master/docs/tutorial.rst)

"""
import logging
from tensorboardX import SummaryWriter


class DLLogger:

    def __init__(self):
        self.handlers = []

    def add_handler(self, handler):
        self.handlers.append(handler)

    def add_scalar(self, tag_name, value, iteration_number):
        """

        :param tag_name: str, should in path format
        :param value:
        :param iteration_number:
        :return:
        """
        for handler in self.handlers:
            handler.add_scalar(tag_name, value, iteration_number)

    def close(self):
        for handler in self.handlers:
            handler.close()


class DLTBHandler(SummaryWriter):
    def __init__(self, path_to_file):
        """

        :param path_to_file: str
        """
        super(DLTBHandler, self).__init__(path_to_file)


class DLLoggingHandler:
    def __init__(self):
        pass

    def add_scalar(self, tag_name, value, iteration_number):
        logging.info('Step: %s \t %s: %.4f', iteration_number, tag_name, value)

    def close(self):
        pass
