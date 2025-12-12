import unittest
import sys
import os
import tensorflow as tf

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.features.feature_builder import build_model

class TestFeatures(unittest.TestCase):
    def test_model_build(self):
        input_shape = (224, 224, 3)
        num_classes = 2
        model = build_model(input_shape, num_classes)
        self.assertIsInstance(model, tf.keras.Model)
        self.assertEqual(model.output_shape, (None, 2))

if __name__ == '__main__':
    unittest.main()
