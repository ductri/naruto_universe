from .. import utils, constants
from ..component import Component


class PreprocessComponent(Component):
    name = constants.PREPROCESS_COMPONENT

    def __init__(self, hparams):
        Component.__init__(self, hparams)
