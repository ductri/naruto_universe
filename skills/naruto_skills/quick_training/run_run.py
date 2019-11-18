from collections import defaultdict
import json

from quick_training.procedure import SimpleProcedure
from quick_training import constants


if __name__ == '__main__':
    hparams = defaultdict(lambda: {})
    hparams[constants.GLOBAL] = dict()
    hparams[constants.GLOBAL][constants.GLOBAL_DIRECTOR] = 'tmp/tmmp/'

    procedure = SimpleProcedure(hparams)
    procedure.train('tmp/new_training_data.csv')
    procedure.save()
    procedure.evaluate()
    # file_name = hparams[constants.GLOBAL][constants.GLOBAL_DIRECTOR] + '/' + hparams[constants.GLOBAL][constants.GLOBAL_HPARAMS]

    file_name = 'tmp/tmmp/hparams.txt'
    with open(file_name, 'rt') as i_f:
        hparams1 = json.load(i_f)
    procedure1 = SimpleProcedure.load_from_hparams(hparams1)
    procedure1.evaluate()
