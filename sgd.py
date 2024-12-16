import numpy as np
from matplotlib import pyplot as plt

import utils
from softmax import Softmax
from utils import batch, LeastSquares

TITLE = "SGD Minimization"


def plot_results(train_losses, test_losses=None, title=TITLE):
    plt.plot(train_losses, label='train', marker="x")
    if test_losses and test_losses is not None:
        plt.plot(test_losses, label='test', marker="o")
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.title(title)
    plt.savefig(f'output\\SGD\\{title}.png')
    plt.show()


def sgd_least_squares_test():
    m = 1000
    n = 100
    X_train, C_train = LeastSquares.gen_data(m, n)
    sgd(LeastSquares(n), X_train, C_train, batch_size=m, title="SGD Minimization In Least Squares")


def sgd_softmax_tests():
    learning_rates = [0.01, 0.1, 0.5, 1]
    batch_sizes = [32, 64, 128]
    data_options = ['GMM', 'Peaks', 'SwissRoll']

    for data_option in data_options:
        X_train, C_train, X_test, C_test = utils.load_data(f'data\\{data_option}Data.mat')
        n, l = X_train.shape[0], C_train.shape[1]

        for learning_rate in learning_rates:
            for batch_size in batch_sizes:
                title = f"Softmax on {data_option} with lr={learning_rate}, batch={batch_size}"
                sgd(Softmax(n, l), X_train, C_train, learning_rate, batch_size, X_test=X_test, C_test=C_test,
                    title=title)


def sgd(train_func, X_train, C_train, lr=0.1, batch_size=32, epochs=100,
        patience=100, X_test=None, C_test=None, title=TITLE):
    also_test = all(val is not None for val in [X_test, C_test])
    epochs_not_improved = 0
    min_train_loss = np.inf

    train_losses = []
    test_losses = []

    for _ in range(epochs):
        # Creating minibatch
        X_batch, C_batch = batch(X_train, C_train, batch_size=batch_size)

        # Updating W
        train_func.train_step(X_batch, C_batch, lr)

        # Calculating loss
        train_loss = train_func.loss(X_train, C_train)
        if np.isnan(train_loss):
            print('nan encountered')
            break
        train_losses.append(train_loss)

        if also_test:
            test_loss = train_func.loss(X_test, C_test)
            if np.isnan(test_loss):
                print('nan encountered')
                break
            test_losses.append(test_loss)

        # Early Stopping
        if train_loss < min_train_loss:
            epochs_not_improved = 0
            min_train_loss = train_loss
        else:
            epochs_not_improved += 1
        if epochs_not_improved > patience:
            break

    plot_results(train_losses, test_losses, title)
