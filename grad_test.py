import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def normalize_vector(v):
    """
    Normalizes a vector to have a norm of 1.

    Parameters:
        v (ndarray): Input vector.

    Returns:
        ndarray: Normalized vector with a norm of 1.
    """
    norm = np.linalg.norm(v)  # Compute the Euclidean norm (L2 norm) of the vector
    if norm == 0:
        raise ValueError("Cannot normalize a zero vector.")
    return v / norm


def gradient_test(loss_func, grad_func, shape, eps=1, num_iters=10, title='Gradient Test'):
    x = np.random.rand(*shape)
    d = normalize_vector(np.random.rand(*shape))

    dt = d.flatten()
    f_x = loss_func(x)
    grad_x = grad_func(x)
    grad_x = grad_x.flatten()
    dt_grad = dt.T @ grad_x
    zero_order = []
    first_order = []
    for i in range(num_iters):
        f_x_d = loss_func(x + eps * d)
        zero_order.append(abs(f_x_d - f_x))
        first_order.append(abs(f_x_d - f_x - eps * dt_grad))
        eps *= 0.5

    results = {'|f(x+ed)-f(x)|': zero_order, '|f(x+ed)-f(x)-ed.T@grad(x)|': first_order}
    plt.figure()
    for name, arr in results.items():
        plt.semilogy(arr, label=name)
    plt.title(title)
    plt.xlabel('iterations')
    plt.legend()
    plt.savefig(f'output/{title}.png')
    plt.show()

    # Exporting data to csv as well
    results = pd.DataFrame(results)
    results.to_csv(f'output/{title}.csv')
