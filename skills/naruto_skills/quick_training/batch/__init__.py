from .. import constants
from ..component import Component


class BatchingComponent(Component):
    name = constants.BATCHING_COMPONENT

    def __init__(self, hparams):
        Component.__init__(self, hparams)
