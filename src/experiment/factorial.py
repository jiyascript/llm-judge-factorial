"""
Config-driven factorial runner.

Turns each underlying (question, resp_a, resp_b) item into the full set of
2^k surface conditions, runs the judge on each, and yields one fully-logged
record per (item, condition). Baseline = the all-control condition
(position=0, formatting=0, length=0); `flipped` is computed against it.

Set INCLUDE_LENGTH = False to run the safe 2x2 (position x formatting) core
first. Flip it to True only after validate_length_preservation looks clean.
"""

import itertools
import json
import time
from pathlib import Path

from src.transformations.transforms import (
    apply_position, apply_formatting, apply_length,
)
from src.judges.judge import slot_to_ab


INCLUDE_LENGTH = False   # start with the guaranteed-clean 2x2; add length later


def _factor_levels():
    factors = {"position": [0, 1], "formatting": [0, 1]}
    if INCLUDE_LENGTH:
        factors["length"] = [0, 1]
    return factors


def _all_conditions():
    factors = _factor_levels()
    keys = list(factors.keys())
    for combo in itertools.product(*[factors[k] for k in keys]):
        yield dict(zip(keys, combo))


def _build_condition(question, resp_a, resp_b, cond):
    """Apply formatting+length to each response, then position to the pair."""
    a = apply_formatting(resp_a, cond.get("formatting", 0))
    b = apply_formatting(resp_b, cond.get("formatting", 0))
    a = apply_length(a, cond.get("length", 0))
    b = apply_length(b, cond.get("length", 0))
    first, second, first_is_a = apply_position(a, b, cond.get("position", 0))
    return first, second, first_is_a


def run_item(judge, item, judge_name):
    """
    item: dict with keys item_id, question, response_a, response_b.
    Yields one record dict per condition. Baseline computed first so `flipped`
    is well-defined for every row.
    """
    # 1) baseline = all controls
    baseline_cond = {k: 0 for k in _factor_levels()}
    first, second, first_is_a = _build_condition(
        item["question"], item["response_a"], item["response_b"], baseline_cond
    )
    res = judge.judge_pair(item["question"], first, second)
    baseline_ab = slot_to_ab(res["slot_verdict"], first_is_a)

    # 2) every condition (including baseline again, for a clean uniform log)
    for cond in _all_conditions():
        first, second, first_is_a = _build_condition(
            item["question"], item["response_a"], item["response_b"], cond
        )
        res = judge.judge_pair(item["question"], first, second)
        verdict_ab = slot_to_ab(res["slot_verdict"], first_is_a)
        flipped = (
            baseline_ab not in ("TIE", "UNPARSED")
            and verdict_ab not in ("TIE", "UNPARSED")
            and verdict_ab != baseline_ab
        )
        yield {
            "item_id": item["item_id"],
            "judge": judge_name,
            "question": item["question"],
            "position": cond.get("position", 0),
            "formatting": cond.get("formatting", 0),
            "length": cond.get("length", 0),
            "first_slot_is_a": first_is_a,
            "slot_verdict": res["slot_verdict"],
            "verdict_ab": verdict_ab,
            "baseline_ab": baseline_ab,
            "flipped": bool(flipped),
            "raw_judge_output": res["raw"],
            "prompt_version": res["prompt_version"],
            "temperature": 0.0,
            "timestamp": time.time(),
        }


def run_experiment(judge, items, judge_name, out_path):
    """
    Streams records to a JSONL file as it goes — so a crash on item 250 keeps
    the first 249. Resume by filtering already-done item_ids.
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        with open(out) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["item_id"])
                except Exception:
                    pass
    with open(out, "a") as f:
        for item in items:
            if item["item_id"] in done:
                continue
            for rec in run_item(judge, item, judge_name):
                f.write(json.dumps(rec) + "\n")
            f.flush()
    return str(out)
