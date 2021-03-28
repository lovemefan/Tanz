# -*- coding: utf-8 -*-
# @Time  : 2021/3/28 11:45
# @Author : lovemefan
# @File : Optimizer.py
import numpy as np


class Optimizer():
    def __init__(self):
        pass

    def gradient_descent(self, df, x, alpha=0.01, iterations: int = 100, epsilon=1e-8) -> list:
        """ 梯度下降
        :param df: 导数函数
        :param x: 输入x
        :param alpha: 学习率
        :param iterations: 迭代次数
        :param epsilon: epsilon > 0, 梯度下降的停止条件
        :return: list: history of x
        """
        assert epsilon > 0, "epsilon must be positive "
        assert alpha > 0, "alpha must be positive "
        assert iterations > 0 and type(iterations) == int, "iterations must be positive integer"

        history = []
        for i in range(iterations):
            if np.max(np.abs(df(x))) < epsilon:
                break
            else:
                x = x - alpha * df(x)
                history.append(x)
        return history

    def momentun(self, df, x, alpha=0.01, gamma=0.8, iterations=100, epsilon=1e-6):
        """Momentum 动量法
        :param df: 导数函数
        :param x: 输入
        :param alpha: 学习率
        :param gamma: 超参数
        :param iterations: 迭代次数
        :param epsilon:
        :return: epsilon > 0, 梯度下降的停止条件
        """

        assert epsilon > 0, "epsilon must be positive "
        assert alpha > 0, "alpha must be positive "
        assert gamma > 0, "gamma must be positive "
        assert iterations > 0 and type(iterations) == int, "iterations must be positive integer"

        history = []
        v = np.zeros_like(x)
        for i in range(iterations):
            if np.max(np.abs(df(x))) < epsilon:
                break
            else:
                v = gamma * v + alpha * df(x)
                x = x - v
                history.append(x)

        return history


if __name__ == '__main__':
    optimizer = Optimizer()
    df = lambda x: np.power(3 * x, 2) - 6 - 10
    result = optimizer.momentun(df, np.array([1.0, 2]), 0.01, 0.8, 20, 1e-8)
