from typing import Tuple

import numpy as np

import utils
from grad_test import gradient_test
from utils import load_data


class Softmax:
    def __init__(self, in_size, out_size):
        self.W = utils.small_rand(in_size, out_size)

    @staticmethod
    def _softmax_calc(z: np.ndarray) -> np.ndarray:
        """
        Computes the softmax of a vector

        Args:
            z: Input vector of weighted sums.

        Returns:
            vector of probabilities for each item in the vector.
        """
        z -= np.max(z, axis=1, keepdims=True)  # for numerical stability
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def loss(self, X: np.ndarray, C: np.ndarray, b=None) -> float:
        return Softmax._calc_loss(X, self.W, C, b)

    @staticmethod
    def _calc_loss(X: np.ndarray, W: np.ndarray, C: np.ndarray, b=None) -> float:
        """
        Computes the loss for softmax regression.

        Args:
            X:  ndarray of shape (n, m)
                Input data (m examples, n features)
            W:  ndarray of shape (n, l)
                Weight matrix (n features, l classes)
            b:  ndarray of shape (1, l)
                Bias vector (1 bias per class)
            C:  ndarray of shape (m, l)
                One-hot encoded matrix of label indicators

        Returns:
            loss: float
                  Multinomial softmax loss value.
        """
        m = X.shape[1]

        # Compute scores
        scores = X.T @ W
        if b is not None:
            scores = scores + b

        probs = Softmax._softmax_calc(scores)

        # compute the loss: average cross-entropy loss
        # get the log of the probability of the true class for each sample
        log_probs = np.log(probs + 1e-15)  # Add small value to avoid log(0)
        return -np.sum(C * log_probs) / m

    @staticmethod
    def grad_X(X, W, C, b=None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes the loss for softmax regression.

        Args:
            X:  ndarray of shape (n, m)
                Input data (m samples, n features)
            W:  ndarray of shape (n, l)
                Weight matrix (n features, l classes)
            b:  ndarray of shape (1, l)
                Bias vector (1 bias per class)
            C:  ndarray of shape (m, l)
                One-hot encoded matrix of label indicators

        Returns:
            grad_X: ndarray of shape (n, m)
                    Gradient of the regression w.r.t the data.
        """
        m = X.shape[1]

        # Compute scores
        scores = X.T @ W
        if b is not None:
            scores = scores + b

        probs = Softmax._softmax_calc(scores)

        # compute the gradient on scores
        return (1 / m) * W @ (probs - C).T  # shape (n, m)

    @staticmethod
    def grad_W(X, W, C, b=None) -> np.ndarray:
        """
        Computes the loss for softmax regression.

        Args:
            X:  ndarray of shape (n, m)
                Input data (m examples, n features)
            W:  ndarray of shape (n, l)
                Weight matrix (n features, l classes)
            C:  ndarray of shape (m, l)
                One-hot encoded matrix of label indicators
            b:  Optional ndarray of shape (l,)
                Bias vector (1 bias per class)

        Returns:
            grad_W: ndarray of shape (n, l)
                    Gradient of the loss w.r.t weights.
        """
        m = X.shape[1]

        # Compute scores
        scores = X.T @ W
        if b is not None:
            scores = scores + b

        # compute the gradient on scores
        probs = Softmax._softmax_calc(scores)  # (m, l)
        return X @ (probs - C) / m  # (n, m) * (m, l) = (n, l)

    def update_weights(self, X, C, learning_rate):
        dW = Softmax.grad_W(X, self.W, C)
        dX = Softmax.grad_X(X, self.W, C)
        self.W -= learning_rate * dW
        return dX

    def train_step(self, X, C, learning_rate):
        Softmax._calc_loss(X, self.W, C)
        return self.update_weights(X, C, learning_rate)


def softmax_gradient_test():
    m = 10
    X, C, _, _ = load_data('data/PeaksData.mat', m)
    loss_func = lambda W: Softmax._calc_loss(X, W, C)
    grad_func = lambda W: Softmax.grad_W(X, W, C)

    gradient_test(loss_func, grad_func, (X.shape[0], C.shape[1]))