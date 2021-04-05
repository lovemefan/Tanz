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

    def gradient_descent_momentun(self, df, x, alpha=0.01, gamma=0.8, iterations=100, epsilon=1e-6):
        """Momentum 动量法,类似于下落物体的惯性，惯性可以像小球一样，当初速度较大时，可以跳出一些比较小的极值区
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

    def gradient_descent_adagrad(self, df, x, alpha=0.01, iterations: int = 100, epsilon=1e-8):
        """Adagrad 自适应梯度，
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
        assert iterations > 0 and type(iterations) == int, "iterations must be positive integer"

        history = []
        gl = np.zeros_like(x)
        for i in range(iterations):
            if np.max(np.abs(df(x))) < epsilon:
                break
            else:
                gl += df(x) ** 2
                v = alpha * df(x) / (np.sqrt(gl) + epsilon)
                x = x - v
                history.append(x)

        return history

    def gradient_descent_Adadelta(self, df, x, alpha=0.1, rho=0.9, iterations=100, epsilon=1e-8):
        """Adagrad 自适应梯度，
        :param df: 导数函数
        :param x: 输入
        :param alpha: 学习率
        :param rho: 平滑参数
        :param iterations: 迭代次数
        :param epsilon:
        :return: epsilon > 0, 梯度下降的停止条件
        """
        assert epsilon > 0, "epsilon must be positive "
        assert alpha > 0, "alpha must be positive "
        assert 0 < rho < 1, "rho must between 0 and 1 "
        assert iterations > 0 and type(iterations) == int, "iterations must be positive integer"

        history = [x]
        eg = np.zeros_like(x)
        Edelta = np.zeros_like(x)

        for i in range(iterations):
            if np.max(np.abs(df(x))) < epsilon:
                break
            else:
                grad = df(x)
                eg = rho * eg + (1 - rho) * grad ** 2
                delta = np.sqrt((Edelta + epsilon) / (eg + epsilon)) * grad
                Edelta = rho * Edelta + (1 - rho) * delta ** 2
                x = x - alpha * delta
                history.append(x)
        return history

    def gradient_descent_RMSprop(self, df, x, alpha=0.1, beta=0.9, iterations=100, epsilon=1e-8):
        """
        :param df: 导数函数
        :param x: 输入
        :param alpha: 学习率
        :param beta: 超参
        :param iterations: 迭代次数
        :param epsilon: epsilon > 0, 梯度下降的停止条件
        :return: list: list of x
        """
        assert epsilon > 0, "epsilon must be positive "
        assert alpha > 0, "alpha must be positive "
        assert 0 < beta < 1, "beta must between 0 and 1  "
        assert iterations > 0 and type(iterations) == int, "iterations must be positive integer"

        history = [x]
        v = np.zeros_like(x)

        for i in range(iterations):
            if np.max(np.abs(df(x))) < epsilon:
                break
            else:
                grad = df(x)
                v = beta * v +(1 - beta) * grad ** 2
                x = x - alpha * grad / (np.sqrt(v) + epsilon)

                history.append(x)
        return history

    def gradient_descent_Adam(self, df, x, alpha=0.1, beta_1=0.9, beta_2=0.99, iterations=100, epsilon=1e-8):
        """
        :param df: 导数函数
        :param x: 输入
        :param alpha: 学习率
        :param beta: 超参
        :param iterations: 迭代次数
        :param epsilon: epsilon > 0, 梯度下降的停止条件
        :return: list: list of x
        """
        assert epsilon > 0, "epsilon must be positive "
        assert alpha > 0, "alpha must be positive "
        assert 0 < beta_1 < 1, "beta1 must between 0 and 1  "
        assert 0 < beta_2 < 1, "beta2 must between 0 and 1  "
        assert iterations > 0 and type(iterations) == int, "iterations must be positive integer"

        history = []
        m = np.zeros_like(x)
        v = np.zeros_like(x)

        for i in range(iterations):
            if np.max(np.abs(df(x))) < epsilon:
                break
            else:
                grad = df(x)
                m = beta_1 * m + (1 - beta_1) * grad
                v = beta_2 * v + (1 - beta_2) * grad ** 2
                m_t = m / (1 - beta_1)
                v_t = v / (1 - beta_2)

                x = x - alpha * m_t / (np.sqrt(v_t) + epsilon)
                history.append(x)
        return history
