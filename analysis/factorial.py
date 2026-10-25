"""2^3 factorial decomposition of first-slot selection rate.

The flip outcome is definitionally zero in the control cell, which makes a full
factorial model on it unidentifiable (this is what caused perfect separation in
the earlier logistic attempt). First-slot selection is well defined in all eight
cells, so the standard factorial contrasts apply directly -- and it is the
quantity the mechanism claim is about.

Effect coding: each factor in {-1,+1}. The contrast for a term is the mean of
(product of its coded factors) x outcome, times 2, giving the change in rate
from turning that factor on. CIs by paired bootstrap over items.
"""
import json
import numpy as np, pandas as pd

D = "../data"
RNG = np.random.default_rng(0)
B = 2000

d = pd.concat([pd.DataFrame([json.loads(l) for l in open(f"{D}/{f}")])
               for f in ("raw_qwen.jsonl", "raw_llama.jsonl")], ignore_index=True)

CELLS = [("control",0,0,0), ("P",1,0,0), ("F",0,1,0), ("L",0,0,1),
         ("PF",1,1,0), ("PL",1,0,1), ("FL",0,1,1), ("PFL",1,1,1)]

TERMS = {"P": ("P",), "F": ("F",), "L": ("L",),
         "P x F": ("P","F"), "P x L": ("P","L"), "F x L": ("F","L"),
         "P x F x L": ("P","F","L")}


def slot_matrix(df):
    """items x 8 cells, 1 if the judge picked the first slot, NaN if TIE/missing."""
    cols = {}
    for name, p, f, l in CELLS:
        s = df[(df.position == p) & (df.formatting == f) & (df.length == l)] \
              .set_index("item_id").slot_verdict
        cols[name] = s.map({"1": 1.0, "2": 0.0})      # TIE -> NaN
    return pd.DataFrame(cols).dropna()


def effects(M):
    """Factorial contrasts on the cell means of M (items x cells)."""
    means = M.mean(axis=0)
    code = {name: {"P": 2*p-1, "F": 2*f-1, "L": 2*l-1}
            for name, p, f, l in CELLS}
    out = {}
    for term, facs in TERMS.items():
        signs = np.array([np.prod([code[n][f] for f in facs]) for n, *_ in CELLS])
        # mean over the 8 cells of sign*rate, x2 -> effect on the rate scale
        out[term] = 2 * float(np.mean(signs * means.values))
    return out


print(f"{'='*72}\n2^3 FACTORIAL DECOMPOSITION OF FIRST-SLOT SELECTION RATE"
      f"\ncontested pairs; effects in percentage points\n{'='*72}")

store = {}
for jkey, jname in (("qwen2.5-7b", "Qwen2.5-7B"), ("llama3.1-8b", "Llama-3.1-8B")):
    M = slot_matrix(d[(d.judge == jkey) & (d.category == "close")])
    base = effects(M)

    idx = np.arange(len(M))
    boot = {t: [] for t in TERMS}
    for _ in range(B):
        take = RNG.choice(idx, size=len(idx), replace=True)
        e = effects(M.iloc[take])
        for t in TERMS:
            boot[t].append(e[t])

    print(f"\n{jname}   (n = {len(M)} items with binary verdicts in all 8 cells)")
    print(f"  grand mean first-slot rate: {M.values.mean()*100:.1f}%")
    print(f"  {'term':<12} {'effect':>8}   {'95% CI':>18}")
    rows = {}
    for t in TERMS:
        lo, hi = np.percentile(boot[t], [2.5, 97.5])
        star = "" if lo < 0 < hi else "  *"
        print(f"  {t:<12} {base[t]*100:+8.1f}   [{lo*100:+6.1f}, {hi*100:+6.1f}]{star}")
        rows[t] = (base[t]*100, lo*100, hi*100)
    store[jname] = rows

    print("  cell means (first-slot %):")
    for name, *_ in CELLS:
        print(f"    {name:<8} {M[name].mean()*100:5.1f}")

pd.DataFrame({j: {t: f"{v[0]:+.1f} [{v[1]:+.1f}, {v[2]:+.1f}]"
                  for t, v in r.items()} for j, r in store.items()}) \
  .to_csv("factorial_effects.csv")
print("\nwrote factorial_effects.csv")
