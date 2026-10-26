import numpy as np
import scipy.sparse as sp
from models.ials import IALS


def build_signal(d, T_end, q90=None, alpha=20.0, tau_days=365.0,
                 use_hours=True, use_sign=True, use_decay=True, min_q90=5.0):
    d = d.sort_values(["u", "i"]).reset_index(drop=True)

    if use_hours:
        if q90 is None:
            q90 = d.groupby("i")["hours"].quantile(0.9)
        q = d["i"].map(q90).fillna(q90.median()).clip(lower=min_q90).to_numpy()
        f = np.minimum(1.0, np.log1p(d["hours"].to_numpy()) / np.log1p(q))
        strength = 0.3 + 0.7 * f
    else:
        strength = np.ones(len(d))

    sign = np.where(d["is_recommended"].to_numpy(), 1.0, -1.0) if use_sign else np.ones(len(d))
    T = sign * strength

    if use_decay:
        age = (T_end - d["date"]).dt.days.to_numpy()
        C = 1.0 + alpha * np.exp(-age / tau_days)
    else:
        C = np.full(len(d), 1.0 + alpha)

    return d, T, C


class CustomIALS:

    def __init__(self, factors=64, reg=300.0, alpha=20.0, tau_days=365.0, epochs=30,
                 use_hours=True, use_sign=True, use_decay=True, seed=42, verbose=False):
        self.kw = dict(alpha=alpha, tau_days=tau_days, use_hours=use_hours,
                       use_sign=use_sign, use_decay=use_decay)
        self.model = IALS(factors=factors, reg=reg, alpha=alpha, epochs=epochs,
                          seed=seed, verbose=verbose)

    def fit(self, seen_df, shape, T_end, eval_fn=None):
        d, T, C = build_signal(seen_df, T_end, **self.kw)
        X = sp.csr_matrix((np.ones(len(d), dtype=np.float32), (d["u"].values, d["i"].values)),
                          shape=shape)
        X.sort_indices()   # порядок data = сортировка по (u, i), как в d
        self.model.fit(X, T=T, C=C, eval_fn=eval_fn)
        return self

    def score(self, users):
        return self.model.score(users)