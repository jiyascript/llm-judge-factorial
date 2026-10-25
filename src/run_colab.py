"""
=====================================================================
COLAB RUNNER  —  judge-compound-bias
Paste these cells into a Colab notebook with a T4 runtime, in order.
This file is NOT meant to be run as a script end-to-end; it documents
the exact sequence. Do the DAY 1 milestone (10 items) before scaling.
=====================================================================
"""

# ---------------------------------------------------------------------
# CELL 1 — install (Colab)
# ---------------------------------------------------------------------
# !pip install -q transformers accelerate bitsandbytes statsmodels pandas
# Then upload/clone this repo so `src/` is importable, or:
# import sys; sys.path.append('/content/judge-compound-bias')

# ---------------------------------------------------------------------
# CELL 2 — load the judge (THE FIRST REAL TEST — verify this runs)
# ---------------------------------------------------------------------
"""
from src.judges.judge import OpenWeightJudge
judge = OpenWeightJudge("Qwen/Qwen2.5-7B-Instruct")   # ~5-6GB in 4-bit on T4

# smoke test — one judgment, printed raw, before anything else:
out = judge.judge_pair(
    "Why does ice float on water?",
    "Ice floats because water expands when it freezes, lowering its density.",
    "Ice is less dense than liquid water, so it floats.",
)
print(out)   # expect {'slot_verdict': '1' or '2' or 'TIE', 'raw': '...'}
# If this errors on load: send me the traceback. If it returns UNPARSED a lot:
# the model isn't following the 'Verdict: N' format — we tune the prompt.
"""

# ---------------------------------------------------------------------
# CELL 3 — load data (start with a tiny hand-made set for DAY 1)
# ---------------------------------------------------------------------
"""
# DAY 1: 10 hand-made items to prove the pipeline. Real dataset comes day 2-3.
items = [
    {
        "item_id": f"demo_{i}",
        "question": q,
        "response_a": a,
        "response_b": b,
    }
    for i, (q, a, b) in enumerate([
        ("Why does ice float on water?",
         "Water expands when it freezes, lowering density, so ice floats.",
         "Ice floats because it is lighter than water."),
        # ... add ~9 more with a clear better answer so flips are meaningful
    ])
]
"""

# For the REAL dataset (day 2-3): load MT-Bench pairwise or an Arena slice
# from HuggingFace datasets, and reshape each row into the item dict above.
# Aim for 200-300 items for the MVP (one judge). Keep item_id stable.

# ---------------------------------------------------------------------
# CELL 4 — DAY 1 MILESTONE: run the 2x2 on 10 items, inspect
# ---------------------------------------------------------------------
"""
from src.experiment.factorial import run_experiment, INCLUDE_LENGTH
print("length factor on?", INCLUDE_LENGTH)   # should be False for the MVP

path = run_experiment(judge, items, judge_name="qwen2.5-7b",
                      out_path="results/raw_qwen.jsonl")

from src.analysis.analyze import load_records, flip_rate_with_ci
df = load_records(path)
print(df[["item_id","position","formatting","verdict_ab","baseline_ab","flipped"]])
print(flip_rate_with_ci(df))
# SANITY CHECKS before scaling:
#  - do any items flip at all? (if zero flips ever, the manipulation is too weak
#    or every pair is too lopsided — pick pairs with closer quality)
#  - how many UNPARSED? (if >5%, fix the prompt/parser before the real run)
"""

# ---------------------------------------------------------------------
# CELL 5 — validate the LENGTH factor BEFORE turning it on
# ---------------------------------------------------------------------
"""
from src.transformations.transforms import apply_length, validate_length_preservation
for it in items[:10]:
    exp = apply_length(it["response_a"], 1)
    print(validate_length_preservation(it["response_a"], exp))
# Every row should show clean=True and new_content_words=[]. If not, the filler
# is leaking content — fix it, or leave INCLUDE_LENGTH=False and ship the 2x2.
# Also EYEBALL ~5 expansions manually; the rule check is necessary not sufficient.
"""

# ---------------------------------------------------------------------
# CELL 6 — full run (day 3-5), then analysis (day 6)
# ---------------------------------------------------------------------
"""
# scale items to the real dataset, rerun run_experiment (it resumes if interrupted)
from src.analysis.analyze import (
    load_records, flip_rate_with_ci, interaction_model, additivity_check
)
df = load_records("results/raw_qwen.jsonl")
print(flip_rate_with_ci(df))            # -> the headline figure
print(additivity_check(df))             # -> super-additive / additive / antagonistic
print(interaction_model(df, judge="qwen2.5-7b"))  # -> beta_FP etc with CIs
"""

# ---------------------------------------------------------------------
# CELL 7 — second judge (STRETCH GOAL, only if day 1-5 went smoothly)
# ---------------------------------------------------------------------
"""
judge2 = OpenWeightJudge("meta-llama/Llama-3.1-8B-Instruct")
run_experiment(judge2, items, judge_name="llama3.1-8b",
               out_path="results/raw_llama.jsonl")
# then concat both jsonl into one df for the cross-judge table
"""
