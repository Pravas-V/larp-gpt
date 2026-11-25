# run the tests using 'PYTHONPATH=src python3 -m unittest discover tests -v'

import random
import unittest
from larp_larp.tensor import Tensor
from tests.gradcheck import gradcheck

def rand_tensor(rows, cols):
    return Tensor([random.uniform(-1, 1) for _ in range(rows * cols)], (rows, cols))

a = rand_tensor(2, 3)
b = rand_tensor(2, 3)
c = rand_tensor(3, 4)

class TestTensor(unittest.TestCase):

    def test_1(self):
        '''(a @ b).sum()'''
        self.assertTrue(gradcheck(lambda x, y: (x + y).sum(), [a, b]))

    def test_2(self):
        '''(a * b).sum()'''
        self.assertTrue(gradcheck(lambda x, y: (x * y).sum(), [a, b]))

    def test_3(self):
        '''(a @ c).sum()'''
        self.assertTrue(gradcheck(lambda x, y: (x @ y).sum(), [a, c]))

    def test_4(self):
        '''a.T.sum()'''
        self.assertTrue(gradcheck(lambda x: (x.T.sum()), [a]))

    def test_5(self):
        '''a.relu().sum()'''
        self.assertTrue(gradcheck(lambda x: (x.relu().sum()), [a]))

    def test_6(self):
        '''(a @ c).relu().sum()'''
        self.assertTrue(gradcheck(lambda x, y: (x @ y).relu().sum(), [a, c]))

    def test_7(self):
        '''(a * a).sum()'''
        self.assertTrue(gradcheck(lambda x: (x * x).sum(), [a]))


if __name__ == '__main__':
    unittest.main()