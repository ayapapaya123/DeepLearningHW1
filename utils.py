import numpy as np
from scipy.io import loadmat


def load_data(fname, m=None):
    # Loading the mat file
    mat = loadmat(fname)
    # Extracting datasets
    X_train, C_train, X_test, C_test = mat['Yt'], mat['Ct'], mat['Yv'], mat['Cv']

    # Choose data randomly
    train_random_perm = np.random.permutation(X_train.shape[1])
    X_train = X_train[:, train_random_perm]
    C_train = C_train[:, train_random_perm]
    test_random_perm = np.random.permutation(X_test.shape[1])
    X_test = X_test[:, test_random_perm]
    C_test = C_test[:, test_random_perm]

    # Limiting dataset size
    if m:
        X_train, C_train, X_test, C_test = X_train[:, :m], C_train[:, :m], X_test[:, :m], C_test[:, :m]

    # Getting shapes right
    C_train, C_test = C_train.T, C_test.T
    return X_train, C_train, X_test, C_test


def batch(X, C, batch_size=32):
    num_of_examples = X.shape[1]
    indices = np.random.choice(num_of_examples, size=batch_size, replace=False)
    X_batch = X[:, indices]
    C_batch = C[indices, :]
    return X_batch, C_batch


def small_rand(*dn):
    significands = np.random.rand(*dn)
    exponents = np.random.randint(-19, -1, dn)
    return significands * 10. ** exponents


class LeastSquares:
    def __init__(self, n):
        self.x = small_rand(n, 1)

    @staticmethod
    def gen_data(m=1000, n=100):
        X = np.random.normal(0, 1, (n, m))
        x_opt = np.random.normal(0, 1, n)
        noise = np.random.normal(0, 0.1, m)
        C = (X.T @ x_opt + noise).reshape((-1, 1))
        return X, C

    def backward(self, X, C):
        C = C.squeeze()
        self.x = self.x.squeeze()
        ret = X @ (X.T @ self.x - C) / C.shape[0]
        return ret.reshape(-1, 1)

    def train_step(self, X, C, learning_rate):
        self.x -= learning_rate * self.backward(X, C)

    def loss(self, X, C):
        return (0.5 / C.shape[0]) * np.linalg.norm(X.T @ self.x - C) ** 2


def flatten(lst):
    if isinstance(lst, (list, tuple, set, range, reversed, np.ndarray)):
        for sub in lst:
            yield from flatten(sub)
    else:
        yield lst


def separate_weights_by_shapes(arr, network_shapes):
    arr = arr.flatten()
    ret = []
    for layer_shapes in network_shapes:
        curr_layer = []
        for weight_shape in layer_shapes:
            # calc how many weights this layer needs
            num_elems = np.prod(weight_shape)
            weight = arr[:num_elems].reshape(weight_shape)
            curr_layer.append(weight)

            # Get rid of weights we already added
            arr = arr[num_elems:]
        ret.append(curr_layer)
    return ret


def validate_accuracy(actual_results, expected_results):
    # Get predicted classes (index of max probability per sample)
    predicted_classes = np.argmax(actual_results, axis=1)

    # Convert one-hot encoded true labels to class indices
    true_classes = np.argmax(expected_results, axis=1)

    # Compute accuracy by comparing predictions to true labels
    return np.mean(predicted_classes == true_classes) * 100