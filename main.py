from activations import Tanh
from networks import neural_net_grad_test, linear_network, residual_network
from layers import LinearLayer, ResidualLayer
from sgd import sgd_least_squares_test, sgd_softmax_tests
from softmax import softmax_gradient_test


def main():
    # softmax_gradient_test()
    # sgd_least_squares_test()
    # sgd_softmax_tests()
    # LinearLayer.jacobian_test()
    # ResidualLayer.jacobian_test()


    neural_net_grad_test(linear_network([2, 5, 5], Tanh), title='Linear Network Gradient Test')
    # neural_net_grad_test(residual_network(2, 5, 2, Tanh), title='Residual Network Gradient Test')


if __name__ == '__main__':
    main()
