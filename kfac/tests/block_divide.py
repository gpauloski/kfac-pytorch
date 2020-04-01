import unittest
import sys

sys.path.append('../')
from kfac_utils import get_block_boundary

class TestBlockDivide(unittest.TestCase):

    def test1(self):
        start, end = get_block_boundary(0, 1, [100, 100])
        self.assertEqual(start, [0, 0])
        self.assertEqual(end, [100, 100])

    def test2(self):
        start, end = get_block_boundary(0, 2, [100, 100])
        self.assertEqual(start, [0, 0])
        self.assertEqual(end, [50, 50])

        start, end = get_block_boundary(1, 2, [100, 100])
        self.assertEqual(start, [50, 50])
        self.assertEqual(end, [100, 100])

    def test3(self):
        start, end = get_block_boundary(0, 3, [100, 100])
        self.assertEqual(start, [0, 0])
        self.assertEqual(end, [33, 33])

        start, end = get_block_boundary(1, 3, [100, 100])
        self.assertEqual(start, [33, 33])
        self.assertEqual(end, [66, 66])

        start, end = get_block_boundary(2, 3, [100, 100])
        self.assertEqual(start, [66, 66])
        self.assertEqual(end, [100, 100])

    def test4(self):
        start, end = get_block_boundary(0, 1, [1, 1])
        self.assertEqual(start, [0, 0])
        self.assertEqual(end, [1, 1])
    
    def test5(self):
        start, end = get_block_boundary(42, 100, [100, 100])
        self.assertEqual(start, [42, 42])
        self.assertEqual(end, [43, 43])

    def test6(self):
        start, end = get_block_boundary(42, 100, [100, 1000])
        self.assertEqual(start, [42, 420])
        self.assertEqual(end, [43, 430])

    def test7(self):
        try:
            start, end = get_block_boundary(100, 100, [100, 1000])
            self.assertTrue(False)
        except ValueError:
            self.assertTrue(True)

    def test8(self):
        try:
            start, end = get_block_boundary(1, 100, [10, 10])
            self.assertTrue(False)
        except ValueError:
            self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()
