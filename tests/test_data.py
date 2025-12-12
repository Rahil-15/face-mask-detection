import os
import sys
import unittest
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.data_loader import load_config, load_data

class TestData(unittest.TestCase):
    def test_config_load(self):
        config = load_config("src/config/config.yaml")
        self.assertTrue(isinstance(config, dict))
        self.assertIn('paths', config)
        self.assertIn('model', config)

    def test_data_paths_exist(self):
        config = load_config("src/config/config.yaml")
        self.assertTrue(os.path.exists(config['paths']['raw_data']))

if __name__ == '__main__':
    unittest.main()
