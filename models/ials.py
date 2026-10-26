import numpy as np
import scipy.sparse as sp


class IALS:
    """iALS

    min  sum_obs c_ui (t_ui - x_u.y_i)^2 + w0 * sum_unobs (x_u.y_i)^2 + reg * (|X|^2 + |Y|^2)
    По умолчанию t_ui = 1, c_ui = 1 + alpha. score(u, i) = x_u . y_i
    """

    def __init__(self, factors=64, reg=10.0, alpha=20.0, w0=1.0,
                 epochs=10, seed=42, verbose=True):
        self.factors, self.reg, self.alpha, self.w0 = factors, reg, alpha, w0
        self.epochs, self.seed, self.verbose = epochs, seed, verbose

    @staticmethod
    def _half_step(indptr, indices, T, C, F, w0, reg):
        f = F.shape[1]
        base = w0 * (F.T @ F) + reg * np.eye(f)           # считается один раз
        out = np.zeros((len(indptr) - 1, f))
        for r in range(len(indptr) - 1):
            s, e = indptr[r], indptr[r + 1]
            if s == e:                                    # нет взаимодействий -> нулевой вектор
                continue
            V = F[indices[s:e]]                           # (n, f)
            c, t = C[s:e], T[s:e]
            A = base + (V.T * (c - w0)) @ V
            out[r] = np.linalg.solve(A, V.T @ (c * t))
        return out

    def fit(self, X, T=None, C=None, eval_fn=None):
        """X: csr (users x items). T, C: массивы длины X.nnz в порядке X.data
        (после X.sort_indices()). eval_fn(model, epoch) вызывается после каждой эпохи."""
        import time
        X = X.tocsr()
        X.sort_indices()
        nnz = X.nnz
        T = np.ones(nnz) if T is None else np.asarray(T, dtype=np.float64)
        C = np.full(nnz, 1.0 + self.alpha) if C is None else np.asarray(C, dtype=np.float64)

        P = sp.csr_matrix((np.arange(1, nnz + 1, dtype=np.int64), X.indices, X.indptr),
                          shape=X.shape).T.tocsr()
        perm = P.data - 1
        Tt, Ct = T[perm], C[perm]

        rng = np.random.default_rng(self.seed)
        n_users, n_items = X.shape
        self.U = np.zeros((n_users, self.factors))
        self.V = rng.normal(0, 0.1, (n_items, self.factors))

        for ep in range(1, self.epochs + 1):
            t0 = time.time()
            self.U = self._half_step(X.indptr, X.indices, T, C, self.V, self.w0, self.reg)
            self.V = self._half_step(P.indptr, P.indices, Tt, Ct, self.U, self.w0, self.reg)
            if self.verbose:
                print(f"эпоха {ep}/{self.epochs}  ({time.time() - t0:.0f} с)", end="")
            if eval_fn is not None:
                eval_fn(self, ep)
            elif self.verbose:
                print()
        return self

    def score(self, users):
        return (self.U[users] @ self.V.T).astype(np.float32)