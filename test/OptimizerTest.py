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
        df = lambda x: 3 * np.power(x, 2) - 9

        self.assertRaises(AssertionError, self.optimizer.gradient_descent, df, 1.0, -0.01, 200, 1e-8)
        self.assertRaises(AssertionError, self.optimizer.gradient_descent, df, 1.0, 0.01, -200, 1e-8)
        self.assertRaises(AssertionError, self.optimizer.gradient_descent, df, 1.0, 0.01, 20.5, 0)
        self.assertRaises(AssertionError, self.optimizer.gradient_descent, df, 1.0, 0.01, 200, -1)

        result = self.optimizer.gradient_descent(df, np.array([1.0, 2]), 0.01, 20, 1e-8)
        assert type(result) == list

    def test_momentum(self):
        df = lambda x: 3 * np.power(x, 2) - 9

        self.assertRaises(AssertionError, self.optimizer.gradient_descent, df, 1.0, -0.01, 200, 1e-8)
        self.assertRaises(AssertionError, self.optimizer.gradient_descent, df, 1.0, 0.01, -200, 1e-8)
        self.assertRaises(AssertionError, self.optimizer.gradient_descent, df, 1.0, 0.01, 20.5, 0)
        self.assertRaises(AssertionError, self.optimizer.gradient_descent, df, 1.0, 0.01, 200, -1)

        result = self.optimizer.gradient_descent_momentun(df, np.array([2.0]), 0.01, 0.8, 500, 1e-8)
        assert np.abs(result[-1] - np.sqrt(3)) < 1e-5

    def test_adagrad(self):
        df = lambda x: 3 * np.power(x, 2) - 9
        result = self.optimizer.gradient_descent_adagrad(df, np.array([2.0]), 0.01, 5000, 1e-8)
        assert np.abs(result[-1] - np.sqrt(3)) < 1e-5

    def test_Adadelta(self):
        df = lambda x: 3 * np.power(x, 2) - 9
        result = self.optimizer.gradient_descent_Adadelta(df, np.array([2.0]), 0.01, 0.9, 50000, 1e-8)
        print(np.abs(result[-1] - np.sqrt(3)))
        assert np.abs(result[-1] - np.sqrt(3)) < 1e-2

    def test_RMSprop(self):
        df = lambda x: 3 * np.power(x, 2) - 9
        result = self.optimizer.gradient_descent_RMSprop(df, np.array([2]), 0.01, 0.9, 200, 1e-8)
        assert np.abs(result[-1] - np.sqrt(3)) < 1e-5

    def test_Adam(self):
        df = lambda x: 3 * np.power(x, 2) - 9
        result = self.optimizer.gradient_descent_Adam(df, np.array([2]), 0.01, 0.9, 0.99, 200, 1e-8)


        assert np.abs(result[-1] - np.sqrt(3)) < 1e-5
