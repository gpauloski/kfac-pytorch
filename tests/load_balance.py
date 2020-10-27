import unittest

from kfac.utils import load_balance


class TestLoadBalance(unittest.TestCase):

    def test1(self):
        n_workers = 1
        work = []
        expected = []
        try:
            result = load_balance(n_workers, work)
            self.assertTrue(False)
        except ValueError as e:
            self.assertTrue(True)
    
    def test2(self):
        n_workers = 1
        work = [1]
        expected = [0]
        result = load_balance(n_workers, work)
        self.assertEqual(result, expected)

    def test3(self):
        n_workers = 1
        work = [1, 2]
        expected = [0, 0]
        result = load_balance(n_workers, work)
        self.assertEqual(result, expected)

    def test4(self):
        n_workers = 2
        work = [1, 2]
        expected = [1, 0]
        result = load_balance(n_workers, work)
        self.assertEqual(result, expected)

    def test5(self):
        n_workers = 2
        work = [1, 1, 2]
        expected = [1, 1, 0]
        result = load_balance(n_workers, work)
        self.assertEqual(result, expected)

    def test6(self):
        n_workers = 2
        work = [1, 1, 1, 1]
        expected = [0, 1, 0, 1]
        result = load_balance(n_workers, work)
        self.assertEqual(result, expected)

    def test7(self):
        n_workers = 3
        work = [1, 1, 1, 1]
        expected = [0, 1, 2, 0]
        result = load_balance(n_workers, work)
        self.assertEqual(result, expected)

    def test8(self):
        n_workers = 3
        work = [5, 8, 5, 12, 5, 7, 6]
        expected = [1, 1, 0, 0, 1, 2, 2]
        result = load_balance(n_workers, work)
        self.assertEqual(result, expected)

if __name__ == '__main__':
    unittest.main()
