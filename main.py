from layers import LinearLayer, ResidualLayer
from sgd import sgd_least_squares_test, sgd_softmax_tests
from softmax import softmax_gradient_test


def main():
    # softmax_gradient_test()
    # sgd_least_squares_test()
    # sgd_softmax_tests()
    # LinearLayer.jacobian_test()
    ResidualLayer.jacobian_test()


if __name__ == '__main__':
    main()
