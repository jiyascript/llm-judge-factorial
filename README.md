# LLM-Judge Surface Sensitivities: A Factorial Study
Code, raw judgments, and analysis for **"Beyond Isolated Biases: How LLM-Judge
Surface Sensitivities Compose."**

LLM judges respond to candidate order, formatting, and response length, and
these sensitivities are almost always measured one factor at a time. Real
responses differ on several at once. This repository runs a full 2³ factorial
over all three on fixed MT-Bench response pairs, holding content constant, and
measures whether isolated effects predict compound behavior.

The short version:

- Length, not position, is the dominant factor on first-slot selection rate
  (−21.1 pp for Qwen2.5-7B, −15.0 pp for Llama-3.1-8B).
- Position has no detectable effect on *how often the first slot wins* — which
  is what a content-blind positional preference predicts.
- A model with no three-way interaction term reproduces the all-three condition
  to within 0.7 pp in both judges.
- Exactly one interaction is detectable (Llama's formatting × length, +4.0 pp),
  and it survives Bonferroni correction across all eight interaction tests.

  ## Layout

```
src/                 experiment code
  judges/judge.py        4-bit open-weight judge wrapper + the prompt
  transformations/       the three perturbations
  experiment/factorial.py  the 2^3 design and runner
  analysis/analyze.py    flip-rate aggregation
  run_colab.py           driver used for the reported runs
notebooks/           Colab drivers with saved outputs, one per run day
data/                raw per-judgment records (one JSON object per line)
analysis/            scripts reproducing every table in the paper
results/             complete-case flip tables
```

## Data

`data/raw_qwen.jsonl` and `data/raw_llama.jsonl` hold one record per judgment:

| field | meaning |
|---|---|
| `item_id` | MT-Bench pair, suffixed by stratum |
| `category` | `close` (contested, n=260), `anchor2` (strong consensus, n=40), `anchor` (superseded) |
| `position`, `formatting`, `length` | factor levels, 0 = control |
| `first_slot_is_a` | whether response A occupies slot 1 in this condition |
| `slot_verdict` | which slot won: `1`, `2`, or `TIE` |
| `verdict_ab` | the same verdict in A/B terms |
| `baseline_ab` | the all-control verdict for this item |
| `flipped` | whether this condition's verdict differs from baseline |
| `raw_judge_output` | the judge's unparsed output |

5,440 records total. **4,800 enter the analyses** — the `anchor` stratum (640
records) comes from a first pass at selecting consensus pairs that admitted
items carrying a single rater judgment, for which unanimity is vacuous. It was
superseded by `anchor2` and is retained here only for completeness.

## Reproducing

All analysis scripts read the JSONL directly and need only `numpy` and `pandas`.
Set `D` at the top of each file to the `data/` directory.

```bash
python analysis/verify.py      # single-factor rates and Table 1
python analysis/factorial.py   # Table 2, factorial effects on first-slot rate
python analysis/slotmodel.py   # Table 3, content-independent reference model
python analysis/threeway.py    # no-three-way prediction and exclusion bounds
python analysis/latent.py      # latent-class exploration (not used; see below)
```

Bootstraps use 2,000 resamples over items with fixed seeds, paired over items to
preserve the repeated-measures structure of the design.

`latent.py` fits a mixture of Bernoulli products to the eight-cell response
patterns. It is included because the result is instructive rather than useful:
the recovered classes largely track which response won rather than any
distinct judge behaviour, so the model does not separate content preference from
slot sensitivity. Reported here so nobody repeats it.

## Re-running the experiment

```bash
pip install -r requirements.txt
```

`transformers` loads Qwen2.5-7B-Instruct and Llama-3.1-8B-Instruct in 4-bit
(nf4) with greedy decoding; the reported runs used a single NVIDIA T4, about an
hour per judge. Llama-3.1 is gated — accept the licence on Hugging Face and
authenticate first. Greedy decoding is deliberate: the study measures verdict
stability under input transformations, so the judge must not add sampling noise.

The judge prompt is in `src/judges/judge.py` and instructs the model to decide
on correctness, relevance and helpfulness and to *ignore differences in length,
formatting, and ordering*.

## Limitations

Two open-weight 7–8B judges; generalisation to frontier API judges is untested.
The strong-consensus stratum is n=40 and underpowered for interaction
estimation. The length perturbation adds no new content words but this does not
establish semantic equivalence — repeated discourse closers may reduce coherence
independently of factual content. Complete-case filtering may introduce
selection bias.


## License

MIT — see [LICENSE](LICENSE).
