# -*- coding: utf-8 -*-
# @Time  : 2021/3/28 12:30
# @Author : lovemefan
# @File : OptimizerTest.py
import unittest
import numpy as np
from optimizer.Optimizer import Optimizer


class OptimizerTest(unittest.TestCase):

    optimizer = Optimizer()

    def test_gradient_descent(self):
        df = lambda x: np.power(3*x, 2) - 6 - 10

        self.assertRaises(AssertionError, self.optimizer.gradient_descent, df, 1.0, -0.01, 200, 1e-8)
        self.assertRaises(AssertionError, self.optimizer.gradient_descent, df, 1.0, 0.01, -200, 1e-8)
        self.assertRaises(AssertionError, self.optimizer.gradient_descent, df, 1.0, 0.01, 20.5, 0)
        self.assertRaises(AssertionError, self.optimizer.gradient_descent, df, 1.0, 0.01, 200, -1)

        result = self.optimizer.gradient_descent(df, np.array([1.0, 2]), 0.01, 20, 1e-8)
        assert type(result) == list
