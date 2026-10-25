"""Does the all-three cell need a three-way term?

Two complementary answers:

(a) PREDICTION. Fit the 2^3 model WITHOUT a three-way term (intercept, three
    main effects, three two-way terms -- 7 parameters) by least squares on the
    eight observed cell means, then compare its prediction for the all-three
    cell against observation. This answers the reviewer's question directly: if
    a no-three-way model lands on the all-three cell, that cell needs no
    three-way component.

(b) EQUIVALENCE BOUND. Report what size of three-way effect the data rule out,
    rather than only that zero was not rejected.
"""
import json
import numpy as np, pandas as pd

D = "../data"
RNG = np.random.default_rng(11)
B = 2000

d = pd.concat([pd.DataFrame([json.loads(l) for l in open(f"{D}/{f}")])
               for f in ("raw_qwen.jsonl", "raw_llama.jsonl")], ignore_index=True)

CELLS = [("control",0,0,0), ("P",1,0,0), ("F",0,1,0), ("L",0,0,1),
         ("PF",1,1,0), ("PL",1,0,1), ("FL",0,1,1), ("PFL",1,1,1)]

# design matrix in effect coding, columns: 1, P, F, L, PF, PL, FL   (no PFL)
def design():
    rows = []
    for _, p, f, l in CELLS:
        xp, xf, xl = 2*p-1, 2*f-1, 2*l-1
        rows.append([1, xp, xf, xl, xp*xf, xp*xl, xf*xl])
    return np.array(rows, dtype=float)

X7 = design()                     # 8 x 7


def slot_matrix(df):
    cols = {}
    for name, p, f, l in CELLS:
        s = df[(df.position == p) & (df.formatting == f) & (df.length == l)] \
              .set_index("item_id").slot_verdict
        cols[name] = s.map({"1": 1.0, "2": 0.0})
    return pd.DataFrame(cols).dropna()


def stats(M):
    """Returns (prediction error at PFL, three-way effect) in rate units."""
    mu = M.mean(axis=0).values
    beta, *_ = np.linalg.lstsq(X7, mu, rcond=None)
    pred = X7 @ beta
    err_pfl = mu[-1] - pred[-1]            # observed - no-3way prediction
    # three-way contrast, same convention as the factorial table
    signs = np.array([ (2*p-1)*(2*f-1)*(2*l-1) for _, p, f, l in CELLS ], dtype=float)
    eff3 = 2 * float(np.mean(signs * mu))
    return err_pfl, eff3, mu[-1], pred[-1]


print("="*76)
print("DOES THE ALL-THREE CELL NEED A THREE-WAY TERM?  (first-slot rate, contested)")
print("="*76)

for jkey, jname in (("qwen2.5-7b", "Qwen2.5-7B"), ("llama3.1-8b", "Llama-3.1-8B")):
    M = slot_matrix(d[(d.judge == jkey) & (d.category == "close")])
    err, eff3, obs, pred = stats(M)

    idx = np.arange(len(M))
    be, b3 = [], []
    for _ in range(B):
        t = RNG.choice(idx, size=len(idx), replace=True)
        e, f3, _, _ = stats(M.iloc[t])
        be.append(e); b3.append(f3)
    be, b3 = np.array(be)*100, np.array(b3)*100

    lo_e, hi_e = np.percentile(be, [2.5, 97.5])
    lo_3, hi_3 = np.percentile(b3, [2.5, 97.5])
    bound = max(abs(lo_3), abs(hi_3))

    print(f"\n{jname}  (n={len(M)})")
    print(f"  observed all-three cell          : {obs*100:5.1f}%")
    print(f"  no-three-way model prediction    : {pred*100:5.1f}%")
    print(f"  prediction error                 : {err*100:+5.2f} pp"
          f"   95% CI [{lo_e:+.2f}, {hi_e:+.2f}]")
    print(f"  three-way effect (table value)   : {eff3*100:+5.2f} pp"
          f"   95% CI [{lo_3:+.2f}, {hi_3:+.2f}]")
    print(f"  --> rules out |three-way| > {bound:.1f} pp at 95%")
