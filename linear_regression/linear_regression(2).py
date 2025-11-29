import numpy as np

def generate_batches(X, y, batch_size):
    """
    param X: np.array[n_objects, n_features] --- матрица объекты-признаки
    param y: np.array[n_objects] --- вектор целевых переменных
    """
    assert len(X) == len(y)
    np.random.seed(42)
    X = np.array(X)
    y = np.array(y)
    perm = np.random.permutation(len(X))

    X_shuffled = X[perm]
    y_shuffled = y[perm]

    for batch_start in range(0, len(X) - batch_size + 1, batch_size):
        batch_end = batch_start + batch_size
        X_batch = X_shuffled[batch_start:batch_end]
        y_batch = y_shuffled[batch_start:batch_end]
        yield X_batch, y_batch