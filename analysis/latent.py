"""Latent-class exploration: are items a mixture of judge 'modes'?

Each item gives an 8-bit vector: did the judge pick the first slot in each
factorial cell? Fit a mixture of products of Bernoullis (latent class model)
by EM, for k = 1, 2, 3. Report BIC, class sizes, per-class cell profiles, and
bootstrap stability of the class assignment.

Identifiability for k classes over 8 binary observations is covered by the
secant-variety results of Allman, Matias & Rhodes (2009); 2-3 classes are
generically identifiable here.
"""
import json, itertools
import numpy as np, pandas as pd

D = "../data"
RNG = np.random.default_rng(0)
CELLS = [("control",0,0,0), ("P",1,0,0), ("F",0,1,0), ("L",0,0,1),
         ("PF",1,1,0), ("PL",1,0,1), ("FL",0,1,1), ("PFL",1,1,1)]
NAMES = [c[0] for c in CELLS]

d = pd.concat([pd.DataFrame([json.loads(l) for l in open(f"{D}/{f}")])
               for f in ("raw_qwen.jsonl", "raw_llama.jsonl")], ignore_index=True)


def slot_matrix(df):
    cols = {}
    for name, p, f, l in CELLS:
        s = df[(df.position == p) & (df.formatting == f) & (df.length == l)] \
              .set_index("item_id").slot_verdict
        cols[name] = s.map({"1": 1.0, "2": 0.0})
    return pd.DataFrame(cols).dropna()


def em(X, k, iters=500, tol=1e-9, seed=0):
    rng = np.random.default_rng(seed)
    n, m = X.shape
    pi = np.full(k, 1.0/k)
    th = np.clip(rng.uniform(0.15, 0.85, size=(k, m)), 1e-3, 1-1e-3)
    ll_old = -np.inf
    for _ in range(iters):
        # E step in log space
        lg = np.array([np.log(pi[j]) + X @ np.log(th[j]) + (1-X) @ np.log(1-th[j])
                       for j in range(k)]).T          # n x k
        mx = lg.max(axis=1, keepdims=True)
        w = np.exp(lg - mx); w /= w.sum(axis=1, keepdims=True)
        ll = float((mx.ravel() + np.log(np.exp(lg - mx).sum(axis=1))).sum())
        # M step
        Nk = w.sum(axis=0) + 1e-12
        pi = Nk / n
        th = np.clip((w.T @ X) / Nk[:, None], 1e-3, 1-1e-3)
        if abs(ll - ll_old) < tol:
            break
        ll_old = ll
    return pi, th, w, ll


def best_em(X, k, restarts=40):
    out = None
    for s in range(restarts):
        r = em(X, k, seed=s)
        if out is None or r[3] > out[3]:
            out = r
    return out


for jkey, jname in (("qwen2.5-7b", "Qwen2.5-7B"), ("llama3.1-8b", "Llama-3.1-8B")):
    X = slot_matrix(d[(d.judge == jkey) & (d.category == "close")]).values
    n, m = X.shape
    print(f"\n{'='*74}\n{jname}   n={n} items, {m} cells\n{'='*74}")

    fits = {}
    for k in (1, 2, 3):
        pi, th, w, ll = best_em(X, k)
        npar = (k - 1) + k*m
        bic = -2*ll + npar*np.log(n)
        fits[k] = (pi, th, w, ll, bic)
        print(f"  k={k}:  logL {ll:9.1f}   params {npar:3d}   BIC {bic:9.1f}")

    kbest = min(fits, key=lambda k: fits[k][4])
    print(f"  -> BIC selects k={kbest}")

    pi, th, w, ll, bic = fits[2]
    order = np.argsort(-th.mean(axis=1))          # label by overall slot-1 tendency
    print(f"\n  k=2 solution (classes ordered by mean first-slot rate):")
    print(f"    {'class':<8} {'size':>6}   " + "  ".join(f"{c:>7}" for c in NAMES))
    for rank, j in enumerate(order):
        print(f"    {rank+1:<8} {pi[j]*100:5.1f}%   " +
              "  ".join(f"{th[j][i]*100:7.1f}" for i in range(m)))

    # how confidently are items assigned?
    conf = w.max(axis=1)
    print(f"\n    items assigned with posterior > 0.9: {(conf>0.9).mean()*100:.0f}%")
    print(f"    mean max-posterior: {conf.mean():.3f}")

    # bootstrap stability of the 2-class split: does the same partition recur?
    hard = w.argmax(axis=1)
    agree = []
    for b in range(100):
        idx = RNG.choice(n, size=n, replace=True)
        pib, thb, wb, llb = best_em(X[idx], 2, restarts=8)
        # match classes to the full-sample fit by profile distance
        d0 = np.abs(thb[:, None, :] - th[None, :, :]).sum(axis=2)
        perm = [int(np.argmin(d0[j])) for j in range(2)]
        if perm[0] == perm[1]:
            continue
        lgb = np.array([np.log(pib[j]) + X @ np.log(thb[j]) + (1-X) @ np.log(1-thb[j])
                        for j in range(2)]).T
        hb = np.array([perm[j] for j in lgb.argmax(axis=1)])
        agree.append((hb == hard).mean())
    print(f"    bootstrap assignment agreement: mean {np.mean(agree)*100:.0f}% "
          f"(5th pct {np.percentile(agree,5)*100:.0f}%)  over {len(agree)} resamples")
