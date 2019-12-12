class Component:
    """A new component"""
    default_hparams = dict()
    name = None
    CONST_DUMPED_FILE = 'dumped_file'

    def __init__(self, hparams=None):
        self.root_hparams = hparams
        self.component_hparams = hparams[self.name]
        self.default_hparams.update(self.component_hparams)
        self.component_hparams.update(self.default_hparams)

        self.is_train = False

    def train(self, training_data, *args, **kwargs):
        """Train this component.

        This is the components chance to train itself provided
        with the training data. The component can rely on
        any context attribute to be present, that gets created
        by a call to :meth:`components.Component.pipeline_init`
        of ANY component and
        on any context attributes created by a call to
        :meth:`components.Component.train`
        of components previous to this one."""
        pass

    def process(self, message, *args, **kwargs):
        """Process an incoming message.

        This is the components chance to process an incoming
        message. The component can rely on
        any context attribute to be present, that gets created
        by a call to :meth:`components.Component.pipeline_init`
        of ANY component and
        on any context attributes created by a call to
        :meth:`components.Component.process`
        of components previous to this one."""
        pass

    def persist(self):
        """Persist this component to disk for future loading."""
        pass

    @classmethod
    def load(cls, component_hparams):
        """Load this component from file."""
        return cls(component_hparams)
