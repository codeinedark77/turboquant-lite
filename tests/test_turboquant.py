import unittest
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class TestTurboquant(unittest.TestCase):
    def test_dirs(self):
        self.assertTrue(os.path.exists('core'))
        self.assertTrue(os.path.exists('inference'))
        self.assertTrue(os.path.exists('kernels'))

if __name__ == '__main__':
    unittest.main()
