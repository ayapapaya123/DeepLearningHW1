import numpy as np
from scipy.io import loadmat


def load_data(fname, m=None):
    # Loading the mat file
    mat = loadmat(fname)
    # Extracting datasets
    X_train = mat['Yt']
    C_train = mat['Ct']
    X_test = mat['Yv']
    C_test = mat['Cv']
    # Shuffling
    # X_train, C_train = shuffle_data(X_train, C_train)
    # X_test, C_test = shuffle_data(X_test, C_test)
    # Limiting dataset size
    if m:
        X_train = X_train[:, :m]
        C_train = C_train[:, :m]
        X_test = X_test[:, :m]
        C_test = C_test[:, :m]
    # Getting shapes right
    C_train = C_train.T
    C_test = C_test.T
    return X_train, C_train, X_test, C_test


def batch(X, C, batch_size=32):
    size = X.shape[1]
    inds = np.random.choice(range(size), batch_size, False)
    inds.sort()
    X_batch = X[:, inds]
    C_batch = C[inds]
    return X_batch, C_batch


def small_rand(*dn):
    significands = np.random.rand(*dn)
    exponents = np.random.randint(-19, -1, dn)
    return significands * 10. ** exponents


class LeastSquares:
    def __init__(self, n):
        self.x = small_rand(n, 1)

    @staticmethod
    def gen_data(m = 1000, n = 100):
        X = np.random.normal(0, 1, (n, m))
        x_opt = np.random.normal(0, 1, n)
        noise = np.random.normal(0, 0.1, m)
        C = X.T @ x_opt + noise
        C = C.reshape((-1, 1))
        return X, C

    def backward(self, X, C):
        C = C.squeeze()
        self.x = self.x.squeeze()
        ret = X @ (X.T @ self.x - C) / C.shape[0]
        ret = ret.reshape(-1, 1)
        return ret

    def train_step(self, X, C, learning_rate):
        self.x -= learning_rate * self.backward(X, C)

    def loss(self, X, C):
        return (0.5 / C.shape[0]) * np.linalg.norm(X.T @ self.x - C) ** 2
