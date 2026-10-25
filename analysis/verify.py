"""Reproduce the paper's reported rates from the raw per-judgment data."""
import json, sys
import pandas as pd, numpy as np

D = "../data"

d = pd.concat([pd.DataFrame([json.loads(l) for l in open(f"{D}/{f}")])
               for f in ("raw_qwen.jsonl", "raw_llama.jsonl")], ignore_index=True)

CELLS = {"control": (0,0,0), "P": (1,0,0), "F": (0,1,0), "L": (0,0,1),
         "PF": (1,1,0), "PL": (1,0,1), "FL": (0,1,1), "PFL": (1,1,1)}

def wide(df):
    """One row per item, one column per factorial cell holding the A/B/TIE verdict."""
    out = {}
    for name, (p, f, l) in CELLS.items():
        s = df[(df.position == p) & (df.formatting == f) & (df.length == l)] \
              .set_index("item_id").verdict_ab
        out[name] = s
    return pd.DataFrame(out)

def flips(w, cells):
    """Binary flip vs. control, complete-case over `cells` + control, binary A/B only."""
    cols = ["control"] + list(cells)
    sub = w[cols].dropna()
    binary = (sub == "A") | (sub == "B")
    sub = sub[binary.all(axis=1)]
    return pd.DataFrame({c: (sub[c] != sub["control"]).astype(int) for c in cells},
                        index=sub.index)

def p_ind(ps):
    """Independent-toggle null: flip iff an odd number of factors toggle."""
    prod = np.prod([1 - 2*p for p in ps])
    return (1 - prod) / 2

JUDGES = {"qwen2.5-7b": "Qwen", "llama3.1-8b": "Llama"}
COMBOS = {"PF": ("P","F"), "PL": ("P","L"), "FL": ("F","L"), "PFL": ("P","F","L")}

for cat, label in (("close", "CONTESTED"), ("anchor2", "STRONG-CONSENSUS")):
    print(f"\n{'='*66}\n{label}  (category={cat})\n{'='*66}")
    for jkey, jname in JUDGES.items():
        w = wide(d[(d.judge == jkey) & (d.category == cat)])
        print(f"\n{jname}")

        # single-factor rates, each on its own complete-case set
        singles = {}
        for s in ("P","F","L"):
            fl = flips(w, (s,))
            singles[s] = fl[s].mean()
            print(f"  {s} alone: {singles[s]*100:5.1f}%   (n={len(fl)})")

        for cname, parts in COMBOS.items():
            fl = flips(w, tuple(parts) + (cname,))
            ps = [fl[p].mean() for p in parts]
            pred = p_ind(ps) * 100
            obs = fl[cname].mean() * 100
            print(f"  {cname:4s} n={len(fl):3d}  pred {pred:5.1f} / obs {obs:5.1f}"
                  f"   diff {obs-pred:+6.1f}")
