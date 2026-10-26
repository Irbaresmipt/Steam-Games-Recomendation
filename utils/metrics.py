import numpy as np
def per_user_ndcg(model, X_seen, Y_rel, cand, users, K=10, bs=2048):
    disc = 1.0 / np.log2(np.arange(2, K + 2))
    ideal = np.cumsum(disc)
    out = []
    for s in range(0, len(users), bs):
        u = users[s:s + bs]
        S = np.asarray(model.score(u), dtype=np.float32)
        S[:, ~cand] = -np.inf
        seen = X_seen[u].tocoo()
        S[seen.row, seen.col] = -np.inf
        top = np.argpartition(-S, K, axis=1)[:, :K]
        order = np.argsort(-np.take_along_axis(S, top, 1), axis=1)
        top = np.take_along_axis(top, order, 1)
        rel = Y_rel[u].toarray() > 0
        hit = np.take_along_axis(rel, top, 1)
        out.append((hit * disc).sum(1) / ideal[np.minimum(rel.sum(1), K) - 1])
    return np.concatenate(out)


def bootstrap_diff(a, b, n=2000, seed=42):
    """Парный бутстрэп по пользователям: разность средних a-b и 95% интервал."""
    rng = np.random.default_rng(seed)
    d = a - b
    m = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)])
    return d.mean(), np.percentile(m, [2.5, 97.5])