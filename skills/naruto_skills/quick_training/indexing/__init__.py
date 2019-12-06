from .. import constants
from ..component import Component


class IndexingComponent(Component):
    name = constants.INDEXING_COMPONENT

    def __init__(self, hparams):
        Component.__init__(self, hparams)
