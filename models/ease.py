import numpy as np


class EASE:
    """EASE (Steck, 2019): ridge-регрессия 'игра по остальным играм' с нулевой диагональю.

    B = I - P / diag(P),  P = (X^T X + lam * I)^-1,  diag(B) = 0
    score(u) = X[u] @ B
    """

    def __init__(self, lam: float = 500.0):
        self.lam = lam

    def fit(self, X):
        G = (X.T @ X).toarray().astype(np.float64)
        G[np.diag_indices_from(G)] += self.lam
        P = np.linalg.inv(G)
        B = -P / np.diag(P)
        np.fill_diagonal(B, 0.0)
        self.B = B.astype(np.float32)
        self.X = X.tocsr()
        return self

    def score(self, users):
        return np.asarray(self.X[users] @ self.B)