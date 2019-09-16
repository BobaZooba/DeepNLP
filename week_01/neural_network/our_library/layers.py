import numpy as np


class Sigmoid:

    def __init__(self):

        self._sigmoid_result = None

    def forward(self, x):

        self._sigmoid_result = 1 / (1 + np.exp(-x))

        return self._sigmoid_result

    def backward(self, grad):

        new_grad = self._sigmoid_result * (1 - self._sigmoid_result) * grad

        return new_grad

    def step(self, learning_step):

        pass


class NLLLoss:

    def __init__(self):

        self._softmax_result = None
        self._y = None

    @staticmethod
    def softmax(x, axis=1):

        exp_scores = np.exp(x)

        return exp_scores / exp_scores.sum(axis, keepdims=True)

    def forward(self, x, y):

        self._softmax_result = self.softmax(x)

        self._y = np.zeros_like(x)
        self._y[np.arange(x.shape[0]), y] = 1

        loss = - (np.log(self._softmax_result) * self._y).sum(1).mean()

        return loss

    def backward(self):

        return (self._softmax_result - self._y) / self._y.shape[0]

    def step(self, learning_rate):

        pass

