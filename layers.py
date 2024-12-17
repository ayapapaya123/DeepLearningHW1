import numpy as np

import activations
import grad_test
import utils


def jacobian_test(calc_func, jac_tmv, x_shape, eps=1, num_iters=10, title="Jacobian Test"):
    x_ = np.random.rand(*x_shape)
    f_x = calc_func(x_)
    u = np.random.rand(*f_x.shape).reshape((-1, 1))

    new_calc_func = lambda x: calc_func(x).squeeze() @ u.squeeze()
    new_grad = lambda x: jac_tmv(x, u)

    grad_test.gradient_test(new_calc_func, new_grad, x_shape, eps=eps, num_iters=num_iters, title=title)


class LinearLayer:
    def __init__(self, in_size, out_size, activation):
        self.W = utils.small_rand(out_size, in_size)
        self.b = utils.small_rand(out_size, 1)
        self.activation = activation

    def set_weights(self, weights):
        self.W = weights

    @staticmethod
    def calc(W, b, X, activation):
        """
        Calculate the result of the linear layer

        :param W: Linear layer weights
        :param b: label biases
        :param X: Input to the layer
        :param activation: non-linear function to act on the result
        :return: Output of the layer
        """
        return activation(W @ X + b)

    @staticmethod
    def calc_grad(W, b, X, V, activation_deriv):
        """
        Helper function for calculating gradients while applied the subsequent gradient
        :param W: Layer weights
        :param b: Layer biases
        :param X: Layer input
        :param V: Gradient from subsequent layer.
        :param activation_deriv: Derivative of the activation function
        :return: Gradient of the weights with respect to the loss.
        """
        return LinearLayer.calc(W, b, X, activation_deriv) * V

    # def update_weights(self, V, learning_rate):
    #     """
    #
    #     :param V: Gradient from subsequent layer
    #     :param learning_rate: hyper param used to modify the weights
    #     :return: Gradient of input which will be propagated back to the previous layer
    #     """
    #     dW = self.backward_W(V)
    #     db = self.backward_b(V)
    #     dX = self.backward_X(V)
    #     self.W -= learning_rate * dW
    #     self.b -= learning_rate * db
    #     return dX

    @staticmethod
    def grad_W(W, b, X, V, activation_deriv):
        """
        Calculate the layer's weights gradient
        :param W: Layer weights
        :param b: Layer biases
        :param X: Layer input
        :param V: Gradient from subsequent layer.
        :param activation_deriv: Derivative of the activation function
        :return: Gradient of the weights with respect to the loss.
        """
        return LinearLayer.calc_grad(W, b, X, V, activation_deriv) @ X.T

    @staticmethod
    def grad_b(W, b, X, V, activation_deriv):
        """
        Calculate the layer's biases gradient
        :param W: Layer weights
        :param b: Layer biases
        :param X: Layer input
        :param V: Gradient from subsequent layer.
        :param activation_deriv: Derivative of the activation function
        :return: Gradient of the weights with respect to the loss.
        """
        return LinearLayer.calc_grad(W, b, X, V, activation_deriv).sum(axis=1).reshape((-1, 1))

    @staticmethod
    def grad_X(W, b, X, V, activation_deriv):
        """
        Calculate the layer's biases gradient
        :param W: Layer weights
        :param b: Layer biases
        :param X: Layer input
        :param V: Gradient from subsequent layer.
        :param activation_deriv: Derivative of the activation function
        :return: Gradient of the weights with respect to the loss.
        """
        return W.T @ LinearLayer.calc_grad(W, b, X, V, activation_deriv)

    @staticmethod
    def jacobian_test():
        X, C, _, _ = utils.load_data('data/PeaksData.mat', 1)
        n, l = X.shape[0], C.shape[1]
        W = np.random.rand(l, n)
        b = np.random.rand(l, 1)
        activation = activations.Tanh

        # Testing W
        jacobian_test(lambda W: LinearLayer.calc(W, b, X, activation.calc),
                      lambda W, V: LinearLayer.grad_W(W, b, X, V, activation.deriv),
                      W.shape, title='Linear Layer Jacobian Test W')
        # Testing b
        jacobian_test(lambda b: LinearLayer.calc(W, b, X, activation.calc),
                      lambda b, V: LinearLayer.grad_b(W, b, X, V, activation.deriv),
                      b.shape, title='Linear Layer Jacobian Test b')
        # Testing X
        jacobian_test(lambda X: LinearLayer.calc(W, b, X, activation.calc),
                      lambda X, V: LinearLayer.grad_X(W, b, X, V, activation.deriv),
                      X.shape, title='Linear Layer Jacobian Test X')


class ResidualLayer:
    def __init__(self, size, activation):
        self.W1 = utils.small_rand(size, size)
        self.W2 = utils.small_rand(size, size)
        self.b = utils.small_rand(size, 1)
        self.activation = activation

    def set_weights(self, W1, W2, b):
        self.W1 = W1
        self.W2 = W2
        self.b = b

    @staticmethod
    def calc(W1, W2, b, X, activation):
        """
        Calculate the result of the linear layer

        :param W1: TODO: explain diff between W1 and W2 in all funcs
        :param b: label biases
        :param X: Input to the layer
        :param activation: non-linear function to act on the result
        :return: Output of the layer
        """
        return X + W2 @ LinearLayer.calc(W1, b, X, activation)

    @staticmethod
    def calc_grad(W1, W2, b, X, V, activation_deriv):
        """
        Helper function for calculating gradients while applied the subsequent gradient
        :param W: Layer weights
        :param b: Layer biases
        :param X: Layer input
        :param V: Gradient from subsequent layer.
        :param activation_deriv: Derivative of the activation function
        :return: Gradient of the weights with respect to the loss.
        """
        return LinearLayer.calc(W1, b, X, activation_deriv) * (W2.T @ V)

    # def update_weights(self, V, learning_rate):
    #     """
    #
    #     :param V: Gradient from subsequent layer
    #     :param learning_rate: hyper param used to modify the weights
    #     :return: Gradient of input which will be propagated back to the previous layer
    #     """
    #     dW = self.backward_W(V)
    #     db = self.backward_b(V)
    #     dX = self.backward_X(V)
    #     self.W -= learning_rate * dW
    #     self.b -= learning_rate * db
    #     return dX

    @staticmethod
    def grad_W1(W1, W2, b, X, V, activation_deriv):
        """
        Calculate the layer's weights gradient
        :param W: Layer weights
        :param b: Layer biases
        :param X: Layer input
        :param V: Gradient from subsequent layer.
        :param activation_deriv: Derivative of the activation function
        :return: Gradient of the weights with respect to the loss.
        """
        return ResidualLayer.calc_grad(W1, W2, b, X, V, activation_deriv) @ X.T

    @staticmethod
    def grad_W2(W1, b, X, V, activation_deriv):
        """
        Calculate the layer's weights gradient
        :param W: Layer weights
        :param b: Layer biases
        :param X: Layer input
        :param V: Gradient from subsequent layer.
        :param activation_deriv: Derivative of the activation function
        :return: Gradient of the weights with respect to the loss.
        """
        return V @ LinearLayer.calc(W1, b, X, activation_deriv).T

    @staticmethod
    def grad_b(W1, W2, b, X, V, activation_deriv):
        """
        Calculate the layer's biases gradient
        :param W: Layer weights
        :param b: Layer biases
        :param X: Layer input
        :param V: Gradient from subsequent layer.
        :param activation_deriv: Derivative of the activation function
        :return: Gradient of the weights with respect to the loss.
        """
        return ResidualLayer.calc_grad(W1, W2, b, X, V, activation_deriv).sum(axis=1).reshape((-1, 1))

    @staticmethod
    def grad_X(W1, W2, b, X, V, activation_deriv):
        """
        Calculate the layer's biases gradient
        :param W: Layer weights
        :param b: Layer biases
        :param X: Layer input
        :param V: Gradient from subsequent layer.
        :param activation_deriv: Derivative of the activation function
        :return: Gradient of the weights with respect to the loss.
        """
        return V + W1.T @ ResidualLayer.calc_grad(W1, W2, b, X, V, activation_deriv)

    @staticmethod
    def jacobian_test():
        X, C, _, _ = utils.load_data('data/PeaksData.mat', 1)
        n = X.shape[0]
        W1 = np.random.rand(n, n)
        W2 = np.random.rand(n, n)
        b = np.random.rand(n, 1)
        activation = activations.Tanh

        # Testing W1
        jacobian_test(lambda W1: ResidualLayer.calc(W1, W2, b, X, activation.calc),
                      lambda W1, V: ResidualLayer.grad_W1(W1, W2, b, X, V, activation.deriv),
                      W1.shape, title='Residual Layer Jacobian Test W1')

        # Testing W2
        jacobian_test(lambda W2: ResidualLayer.calc(W1, W2, b, X, activation.calc),
                      lambda _, V: ResidualLayer.grad_W2(W1, b, X, V, activation.deriv),
                      W1.shape, title='Residual Layer Jacobian Test W2')

        # Testing b
        jacobian_test(lambda b: ResidualLayer.calc(W1, W2, b, X, activation.calc),
                      lambda b, V: ResidualLayer.grad_b(W1, W2, b, X, V, activation.deriv),
                      b.shape, title='Residual Layer Jacobian Test b')
        # Testing X
        jacobian_test(lambda X: ResidualLayer.calc(W1, W2, b, X, activation.calc),
                      lambda X, V: ResidualLayer.grad_X(W1, W2, b, X, V, activation.deriv),
                      X.shape, title='Residual Layer Jacobian Test X')