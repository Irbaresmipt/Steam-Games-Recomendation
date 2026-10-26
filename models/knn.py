import numpy as np
import scipy.sparse as sp


class ItemKNN:

    def __init__(self, k: int = 100, beta: float = 10.0):
        self.k = k
        self.beta = beta

    def fit(self, X):
        X = X.tocsr().astype(np.float32)
        C = (X.T @ X).toarray()                      # общие пользователи, items x items
        n = np.asarray(X.sum(0)).ravel()
        denom = np.sqrt(np.outer(n, n)) + self.beta
        C = np.divide(C, denom, out=np.zeros_like(C), where=denom > 0)
        np.fill_diagonal(C, 0.0)

        # оставляем top-k в каждом столбце (соседи игры j)
        k = min(self.k, C.shape[0] - 1)
        idx = np.argpartition(-C, k, axis=0)[:k]     # индексы top-k по каждому столбцу
        keep = np.zeros_like(C, dtype=bool)
        np.put_along_axis(keep, idx, True, axis=0)
        C[~keep] = 0.0

        self.S = sp.csr_matrix(C)
        self.X = X
        return self

    def score(self, users):
        return np.asarray((self.X[users] @ self.S).todense())