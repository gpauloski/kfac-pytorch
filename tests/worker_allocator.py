import argparse
import unittest

from kfac.utils import WorkerAllocator


class TestLoadBalance(unittest.TestCase):

    def test1(self):
        self.assertEqual(
            WorkerAllocator.partition_grad_ranks(9, 1.0),
            [[0, 1, 2, 3, 4, 5, 6, 7, 8]]
        )
        self.assertEqual(
            WorkerAllocator.partition_grad_ranks(9, 0.0),
            [[0], [1], [2], [3], [4], [5], [6], [7], [8]]
        )
        self.assertEqual(
            WorkerAllocator.partition_grad_ranks(9, 1/3),
            [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        )
        self.assertEqual(
            WorkerAllocator.partition_grad_ranks(8, 1/3),
            [[0, 1, 2], [3, 4, 5], [6, 7]]
        )
        self.assertEqual(
            WorkerAllocator.partition_grad_ranks(1, 1/3),
            [[0]]
        )
    
    def test2(self):
        self.assertEqual(
            WorkerAllocator.partition_inv_ranks(9, 1.0),
            [[0], [1], [2], [3], [4], [5], [6], [7], [8]]
        )
        self.assertEqual(
            WorkerAllocator.partition_inv_ranks(9, 0.0),
            [[0, 1, 2, 3, 4, 5, 6, 7, 8]]
        )
        self.assertEqual(
            WorkerAllocator.partition_inv_ranks(9, 1/3),
            [[0, 3, 6], [1, 4, 7], [2, 5, 8]]
        )
        self.assertEqual(
            WorkerAllocator.partition_inv_ranks(8, 1/3),
            [[0, 3, 6], [1, 4, 7], [2, 5]]
        )
        self.assertEqual(
            WorkerAllocator.partition_inv_ranks(1, 1/3),
            [[0]]
        )
            
    def test3(self):
        w = WorkerAllocator(9, 0.0)
        for i in range(9):
            self.assertEqual(None, w.get_inv_group(i))
            self.assertEqual(None, w.get_grad_group(i))

    def test4(self):
        w = WorkerAllocator(9, 1)
        for i in range(9):
            self.assertEqual(None, w.get_inv_group(i))
            self.assertEqual(None, w.get_grad_group(i))

if __name__ == '__main__':
    unittest.main()

