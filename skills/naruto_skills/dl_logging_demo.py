import logging
import numpy as np

from naruto_skills.dl_logging import DLLogger, DLTBHandler, DLLoggingHandler


def loss():
    while True:
        yield np.random.rand()


loss_gen = loss()

if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)
    my_logger = DLLogger()
    my_logger.add_handler(DLTBHandler('tmp/exp_1'))
    my_logger.add_handler(DLLoggingHandler())
    interval = 100

    for step in range(10000):
        if step % interval == 0:
            my_logger.add_scalar('Loss/loss1', next(loss_gen), step)
            my_logger.add_scalar('Loss/loss2', next(loss_gen), step)
            my_logger.add_scalar('Loss/loss3', next(loss_gen), step)

    my_logger.close()

    my_logger = DLLogger()
    my_logger.add_handler(DLTBHandler('tmp/exp_2'))
    my_logger.add_handler(DLLoggingHandler())
    interval = 100

    for step in range(10000):
        if step % interval == 0:
            my_logger.add_scalar('Loss/loss1', next(loss_gen), step)
            my_logger.add_scalar('Loss/loss2', next(loss_gen), step)
            my_logger.add_scalar('Loss/loss3', next(loss_gen), step)

    my_logger.close()
