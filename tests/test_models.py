import unittest
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class TestModelScripts(unittest.TestCase):
    def test_imports(self):
        try:
            import src.train
            import src.evaluate
            import src.predict
        except ImportError:
            self.fail("Could not import model scripts")

if __name__ == '__main__':
    unittest.main()
