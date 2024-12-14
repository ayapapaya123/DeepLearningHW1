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
