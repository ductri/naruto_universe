from collections import defaultdict
import unittest
from unittest import TestCase

import torch

from naruto_skills.quick_training.magic.cnn_family import SimpleCNN
from naruto_skills.quick_training import constants


class MagicModelTest(TestCase):

    def test_valid(self):
        hparams_root = defaultdict(lambda: {})
        hparams_root[constants.GLOBAL][constants.GLOBAL_SEQUENCE_LENGTH] = 100
        hparams = hparams_root[constants.MAGIC_COMPONENT]
        hparams['vocab_size'] = 100
        hparams['num_classes'] = 3

        model = SimpleCNN(hparams_root)
        tensor_input = torch.randint(high=100, size=(20, 100), dtype=torch.long)
        tensor_output = model._forward(tensor_input)

        self.assertListEqual(list(tensor_output.size()), [20, 3])


if __name__ == '__main__':
    unittest.main()
