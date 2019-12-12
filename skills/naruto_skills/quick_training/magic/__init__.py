from .. import constants
from ..component import Component


class MagicComponent(Component):
    name = constants.MAGIC_COMPONENT

    def __init__(self, hparams):
        Component.__init__(self, hparams)

    def fit(self, data_loader, *args, **kwargs):
        pass

