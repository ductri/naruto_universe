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
        self.__current_step = None
        self.__temp = []
        self.__filters = []

    def add_scalar(self, tag_name, value, iteration_number):
        if iteration_number != self.__current_step:
            text_render = ['%s: %.4f' % (tag_name, value) for (tag_name, value) in self.__temp]
            text_render = '\t'.join(text_render)

            logging.info('Step: %s \t %s', self.__current_step, text_render)
            self.__temp = []
            self.__current_step = iteration_number

        self.__temp.append((tag_name, value))

    def close(self):
        pass


cache = {}


def get_logger_instance(name):
    global cache
    if name in cache:
        return cache[name]
    else:
        cache[name] = DLLogger()
        return cache[name]
