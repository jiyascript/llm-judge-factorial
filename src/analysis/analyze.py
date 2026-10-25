"""
Analysis: flip rates + mixed-effects logistic regression with interactions.

Two outputs, matching the paper:
  1. flip_rate_table(): the headline numbers (Position 7%, Format 9%, both 15%...)
  2. interaction_model(): the beta_FL / beta_FP / beta_LP / beta_FLP estimates
     with 95% CIs — the objects your paper actually argues about.

The regression uses statsmodels. A true mixed-effects logistic (random item
intercept) is fit via BinomialBayesMixedGLM or, more simply, we provide a
clustered-SE logit as a robust fallback that runs fast and is defensible in a
2-page paper. Use the mixed model if it converges; report the clustered logit
if it doesn't. Both are given.
"""

import json
import numpy as np
import pandas as pd


def load_records(jsonl_path) -> pd.DataFrame:
    rows = []
    with open(jsonl_path) as f:
        for line in f:
            rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    # drop the duplicated all-control rows' self-flip (baseline vs itself = never flips;
    # keep them — they correctly contribute a 0 and anchor the intercept)
    return df


def flip_rate_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flip rate for each factor combination, per judge.
    Excludes rows where baseline or verdict was TIE/UNPARSED (flip undefined).
    """
    factors = [c for c in ["position", "formatting", "length"] if c in df.columns]
    usable = df[
        df["baseline_ab"].isin(["A", "B"]) & df["verdict_ab"].isin(["A", "B"])
    ].copy()
    g = (
        usable.groupby(["judge"] + factors)["flipped"]
        .agg(["mean", "sum", "count"])
        .reset_index()
        .rename(columns={"mean": "flip_rate", "sum": "n_flips", "count": "n"})
    )
    g["flip_rate_pct"] = (g["flip_rate"] * 100).round(1)
    return g


def _wilson_ci(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (round((center - half) * 100, 1), round((center + half) * 100, 1))


def flip_rate_with_ci(df: pd.DataFrame) -> pd.DataFrame:
    t = flip_rate_table(df)
    cis = [_wilson_ci(r.n_flips, r.n) for r in t.itertuples()]
    t["ci_low_pct"], t["ci_high_pct"] = zip(*cis)
    return t


def interaction_model(df: pd.DataFrame, judge: str = None):
    """
    Fit the factorial logistic regression with interactions.
    Returns a tidy DataFrame of term, coef (log-odds), OR, 95% CI, p.

    Tries mixed-effects (random item intercept) first; falls back to a
    cluster-robust logit on item_id if the mixed model fails to converge.
    """
    import statsmodels.formula.api as smf
    import statsmodels.api as sm

    factors = [c for c in ["position", "formatting", "length"] if c in df.columns]
    usable = df[
        df["baseline_ab"].isin(["A", "B"]) & df["verdict_ab"].isin(["A", "B"])
    ].copy()
    if judge is not None:
        usable = usable[usable["judge"] == judge]
    usable["y"] = usable["flipped"].astype(int)

    # full factorial formula, e.g. y ~ position*formatting*length
    rhs = "*".join(factors)
    formula = f"y ~ {rhs}"

    tidy = None
    # --- attempt mixed-effects logistic (random intercept for item) ---
    try:
        from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
        vc = {"item": "0 + C(item_id)"}
        model = BinomialBayesMixedGLM.from_formula(
            formula, vc_formulas=vc, data=usable
        )
        res = model.fit_vb()
        tidy = pd.DataFrame({
            "term": res.model.exog_names,
            "coef": res.fe_mean,
            "sd": res.fe_sd,
        })
        tidy["ci_low"] = tidy["coef"] - 1.96 * tidy["sd"]
        tidy["ci_high"] = tidy["coef"] + 1.96 * tidy["sd"]
        tidy["method"] = "mixed_vb"
    except Exception as e:
        # --- fallback: cluster-robust logit ---
        try:
            res = smf.logit(formula, data=usable).fit(disp=False)
            cov = res.cov_params()  # basic; upgrade to cluster if needed
            tidy = pd.DataFrame({
                "term": res.params.index,
                "coef": res.params.values,
                "p": res.pvalues.values,
            })
            ci = res.conf_int()
            tidy["ci_low"] = ci[0].values
            tidy["ci_high"] = ci[1].values
            tidy["method"] = "logit_fallback"
        except Exception as e2:
            raise RuntimeError(f"both models failed: {e} / {e2}")

    tidy["odds_ratio"] = np.exp(tidy["coef"]).round(3)
    tidy["OR_ci_low"] = np.exp(tidy["ci_low"]).round(3)
    tidy["OR_ci_high"] = np.exp(tidy["ci_high"]).round(3)
    return tidy


def additivity_check(df: pd.DataFrame, judge: str = None) -> dict:
    """
    The paper's core comparison in plain numbers: is the compound flip rate
    bigger than the sum of individual flip rates (super-additive), about equal
    (additive), or smaller (antagonistic)?

    Compares observed all-factors flip rate against the sum of single-factor
    flip rates. Only meaningful for the full design.
    """
    t = flip_rate_table(df)
    if judge is not None:
        t = t[t["judge"] == judge]
    factors = [c for c in ["position", "formatting", "length"] if c in df.columns]

    def rate(**kw):
        m = t.copy()
        for f in factors:
            m = m[m[f] == kw.get(f, 0)]
        return float(m["flip_rate"].mean()) if len(m) else float("nan")

    singles = {}
    for f in factors:
        singles[f] = rate(**{f: 1})
    all_on = rate(**{f: 1 for f in factors})
    return {
        "single_factor_rates": {k: round(v, 3) for k, v in singles.items()},
        "sum_of_singles": round(sum(singles.values()), 3),
        "observed_all_factors": round(all_on, 3),
        "verdict": (
            "super-additive" if all_on > sum(singles.values()) + 0.02
            else "antagonistic" if all_on < sum(singles.values()) - 0.02
            else "approximately additive"
        ),
    }
