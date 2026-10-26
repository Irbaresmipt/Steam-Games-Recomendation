import numpy as np
import torch
import torch.nn.functional as Fn

ACT_EDGES = [1, 2, 3, 5, 8, 13, 21, 50]


class FM:

    def __init__(self, item_feats, use_meta=True, factors=32, lr=5e-3, reg=1e-3,
                 epochs=10, batch=4096, n_neg=3, seed=42, verbose=True):
        self.use_meta, self.k, self.lr, self.reg = use_meta, factors, lr, reg
        self.epochs, self.batch, self.n_neg = epochs, batch, n_neg
        self.seed, self.verbose = seed, verbose
        F = item_feats.toarray().astype(np.float32)
        self.F = torch.from_numpy(F if use_meta else np.zeros_like(F))
        self.m = 1.0 if use_meta else 0.0
        self.edges = torch.tensor(ACT_EDGES, dtype=torch.float32)

    def _item_side(self, items):
        Fi = self.F[items]
        vid = self.V_iid[items]
        b = vid + Fi @ self.V_f
        sq = (vid ** 2).sum(1) + Fi @ (self.V_f ** 2).sum(1)
        inner = 0.5 * ((b ** 2).sum(1) - sq)
        lin = self.w_iid[items] + Fi @ self.w_f
        return b, lin + inner

    def _user_side(self, u, prof, n):
        act = torch.bucketize(n, self.edges, right=True)
        return self.V_uid[u] + self.m * self.V_act[act] + (self.m * prof) @ self.V_prof

    def _sample_neg(self, u):
        u = u.numpy().astype(np.int64)
        neg = self.rng.choice(self.pool, size=(len(u), self.n_neg))
        for _ in range(3):
            key = u[:, None] * self.n_items + neg
            pos = np.minimum(np.searchsorted(self.keys, key), len(self.keys) - 1)
            hit = self.keys[pos] == key
            if not hit.any():
                break
            neg[hit] = self.rng.choice(self.pool, size=int(hit.sum()))
        return torch.from_numpy(neg)

    @torch.no_grad()
    def _finalize(self):
        self.IB, self.IS = self._item_side(torch.arange(self.n_items))

    def fit(self, X, eval_fn=None):
        """X: csr (users x items), видимая часть. eval_fn(model, epoch) после каждой эпохи."""
        torch.manual_seed(self.seed)
        self.rng = np.random.default_rng(self.seed)
        X = X.tocsr()
        self.n_users, self.n_items = X.shape
        k, nf = self.k, self.F.shape[1]

        self.P = torch.from_numpy(np.asarray(X @ self.F.numpy(), dtype=np.float32))
        self.n = torch.from_numpy(np.asarray(X.sum(1)).ravel().astype(np.float32))
        coo = X.tocoo()
        pu = torch.from_numpy(coo.row.astype(np.int64))
        pi = torch.from_numpy(coo.col.astype(np.int64))
        self.keys = np.sort(coo.row.astype(np.int64) * self.n_items + coo.col)
        self.pool = np.where(np.asarray(X.sum(0)).ravel() > 0)[0]

        def P_(*shape):
            return torch.nn.Parameter(torch.randn(*shape) * 0.05)
        self.V_uid, self.V_act = P_(self.n_users, k), P_(len(ACT_EDGES) + 1, k)
        self.V_prof, self.V_iid, self.V_f = P_(nf, k), P_(self.n_items, k), P_(nf, k)
        self.w_iid = torch.nn.Parameter(torch.zeros(self.n_items))
        self.w_f = torch.nn.Parameter(torch.zeros(nf))
        params = [self.V_uid, self.V_act, self.V_prof, self.V_iid, self.V_f, self.w_iid, self.w_f]
        opt = torch.optim.Adam(params, lr=self.lr)

        for ep in range(1, self.epochs + 1):
            perm = torch.randperm(len(pu))
            tot, steps = 0.0, 0
            for s in range(0, len(perm), self.batch):
                idx = perm[s:s + self.batch]
                u, i = pu[idx], pi[idx]
                B = len(u)
                nu = self.n[u]
                prof = (self.P[u] - self.F[i]) / (nu - 1).clamp(min=1).unsqueeze(1)
                a = self._user_side(u, prof, nu - 1)
                neg = self._sample_neg(u.cpu())
                bi, si = self._item_side(i)
                bn, sn = self._item_side(neg.reshape(-1))
                pos_s = si + (a * bi).sum(1)
                neg_s = sn.view(B, -1) + (a.unsqueeze(1) * bn.view(B, self.n_neg, -1)).sum(2)
                loss = -Fn.logsigmoid(pos_s.unsqueeze(1) - neg_s).mean()
                loss = loss + self.reg * ((a ** 2).sum(1).mean() + (bi ** 2).sum(1).mean()
                                          + (bn ** 2).sum(1).mean())
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += loss.item()
                steps += 1
            if self.verbose:
                print(f"эпоха {ep}/{self.epochs}  loss={tot / steps:.4f}", end="")
            self._finalize()
            if eval_fn is not None:
                eval_fn(self, ep)
            elif self.verbose:
                print()
        return self

    @torch.no_grad()
    def score(self, users):
        u = torch.as_tensor(np.asarray(users), dtype=torch.long)
        nu = self.n[u]
        prof = self.P[u] / nu.clamp(min=1).unsqueeze(1)
        a = self._user_side(u, prof, nu)
        return (self.IS.unsqueeze(0) + a @ self.IB.T).numpy().astype(np.float32)