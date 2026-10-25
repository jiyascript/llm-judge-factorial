"""How much of each compound flip rate does a pure slot-preference model explain?

Model: in condition c the judge picks the first slot with probability p_c,
independently of which response sits there and independently of its own choice
in the control condition. Then, writing s_c = 1 if the condition swaps order:

    P(verdict_c != verdict_control)
        = p_ctrl (1 - p_c) + (1 - p_ctrl) p_c          if not swapped
        = p_ctrl p_c       + (1 - p_ctrl)(1 - p_c)     if swapped

p_ctrl and p_c are measured, not fitted. The gap between this prediction and the
observed flip rate is the part the slot prior does NOT explain -- i.e. the
content-dependent correlation between a judge's two verdicts.
"""
import json
import numpy as np, pandas as pd

D = "../data"
RNG = np.random.default_rng(1)
B = 2000

d = pd.concat([pd.DataFrame([json.loads(l) for l in open(f"{D}/{f}")])
               for f in ("raw_qwen.jsonl", "raw_llama.jsonl")], ignore_index=True)

CELLS = [("control",0,0,0), ("P",1,0,0), ("F",0,1,0), ("L",0,0,1),
         ("PF",1,1,0), ("PL",1,0,1), ("FL",0,1,1), ("PFL",1,1,1)]
SWAPPED = {n: bool(p) for n, p, f, l in CELLS}


def matrices(df):
    slot, ab = {}, {}
    for name, p, f, l in CELLS:
        s = df[(df.position == p) & (df.formatting == f) & (df.length == l)].set_index("item_id")
        slot[name] = s.slot_verdict.map({"1": 1.0, "2": 0.0})
        ab[name] = s.verdict_ab
    S = pd.DataFrame(slot); A = pd.DataFrame(ab)
    keep = S.dropna().index
    return S.loc[keep], A.loc[keep]


def predict_obs(S, A):
    p = S.mean(axis=0)
    out = {}
    for name, *_ in CELLS:
        if name == "control":
            continue
        pc, p0 = p[name], p["control"]
        pred = (p0*pc + (1-p0)*(1-pc)) if SWAPPED[name] else (p0*(1-pc) + (1-p0)*pc)
        obs = (A[name] != A["control"]).mean()
        out[name] = (pred, obs)
    return out


print(f"{'='*78}\nSLOT-PREFERENCE MODEL vs OBSERVED FLIP RATE  (contested pairs)\n"
      f"predicted from measured first-slot rates alone; no fitted parameters\n{'='*78}")

for jkey, jname in (("qwen2.5-7b", "Qwen2.5-7B"), ("llama3.1-8b", "Llama-3.1-8B")):
    S, A = matrices(d[(d.judge == jkey) & (d.category == "close")])
    base = predict_obs(S, A)

    idx = np.arange(len(S))
    boot = {c: [] for c in base}
    for _ in range(B):
        t = RNG.choice(idx, size=len(idx), replace=True)
        pr = predict_obs(S.iloc[t], A.iloc[t])
        for c, (pd_, ob) in pr.items():
            boot[c].append(ob - pd_)

    print(f"\n{jname}  (n = {len(S)} items)   first-slot rate, control = {S['control'].mean()*100:.1f}%")
    print(f"  {'cond':<6} {'slot-model':>11} {'observed':>10} {'residual':>10}   {'95% CI on residual':>22}")
    for c in ("P","F","L","PF","PL","FL","PFL"):
        pr, ob = base[c]
        lo, hi = np.percentile(boot[c], [2.5, 97.5])
        star = "" if lo < 0 < hi else " *"
        print(f"  {c:<6} {pr*100:10.1f} {ob*100:10.1f} {(ob-pr)*100:+10.1f}"
              f"   [{lo*100:+6.1f}, {hi*100:+6.1f}]{star}")

    # share of the all-three flip rate attributable to the slot shift
    pr, ob = base["PFL"]
    print(f"  --> slot model accounts for {pr/ob*100:.0f}% of the observed "
          f"all-three flip rate ({ob*100:.1f}%)")
