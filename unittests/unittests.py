import unittest
import torch
import numpy as np
import os
import sys

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(ROOT_DIR))
CONF_FILE = os.getenv('CONF_PATH')

from training.train_script import IrisNet 

class TestIrisNet(unittest.TestCase):

    def setUp(self):
        self.model = IrisNet()

    def test_initialization(self):
        """Test whether the model initializes correctly."""
        self.assertEqual(len(list(self.model.parameters())), 4)  # Check number of parameter tensors

    def test_forward_pass(self):
        """Test the forward pass of the model."""
        sample_input = torch.rand(1, 4)  # Create a random sample input
        output = self.model(sample_input)
        self.assertEqual(output.shape, (1, 3))  # Check output shape

if __name__ == '__main__':
    unittest.main()