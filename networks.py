import numpy as np

import activations
import layers
import utils
from grad_test import gradient_test
from softmax import Softmax


def neural_net_grad_test(net, m=100, title="NN Gradient Test"):
    X, C, _, _ = utils.load_data('data/PeaksData.mat', m)
    gradient_test(lambda weights: net.grad_test_forward(weights, X, C),
                  lambda weights: net.grad_test_backwards(weights, X, C),
                  (net.get_num_weights(),), title=title)


class SequentialNetwork:
    def __init__(self, layers, out_layer):
        self.layers = layers
        self.out_layer = out_layer

    def get_weight_shapes(self):
        shapes = [layer.get_weight_shapes() for layer in self.layers]
        shapes.append(self.out_layer.get_weight_shapes())
        return shapes

    def get_num_weights(self):
        return sum(sum(np.prod(weights_shape)
                       for weights_shape in layer_shapes)
                   for layer_shapes in self.get_weight_shapes())

    def forward(self, X, C):
        for layer in self.layers:
            X = layer.forward(X)
        return self.out_layer.loss(X, C)

    def loss(self, X, C):
        return self.forward(X, C)

    def update_weights(self, learning_rate):
        V = self.out_layer.update_weights(learning_rate)
        for layer in reversed(self.layers):
            V = layer.update_weights(V, learning_rate)

    def backward_weights(self, X, C):
        ret = [self.out_layer.backward_weights()]
        V = self.out_layer.backwards_X()

        for layer in reversed(self.layers):
            ret.append(layer.backward_weights(V))
            V = layer.backward_X(V)
        return list(reversed(ret))

    def train_step(self, X, C, learning_rate):
        self.loss(X, C)
        self.update_weights(learning_rate)

    def grad_test_func(self, func, weights, X, C):
        weights = utils.unflatten_numpy_array(weights, self.get_weight_shapes())
        for layer, weight in zip(self.layers, weights[:-1]):
            layer.set_weights(weight)
        self.out_layer.set_weights(weights[-1])
        ret = func(X, C)
        ret = utils.flatten(ret)
        ret = list(ret)
        ret = np.array(ret)
        return ret

    def grad_test_forward(self, weights, X, C):
        return self.grad_test_func(self.forward, weights, X, C)

    def grad_test_backwards(self, weights, X, C):
        return self.grad_test_func(self.backward_weights, weights, X, C)


def linear_network(sizes, activation):
    return SequentialNetwork([layers.LinearLayer(sizes[i], sizes[i + 1], activation) for i in range(len(sizes) - 2)],
                             Softmax(sizes[-2], sizes[-1]))


def residual_network(in_size, out_size, L, activation):
    return SequentialNetwork([layers.ResidualLayer(in_size, activation)] * L,
                             Softmax(in_size, out_size))