"""
Open-weight judge wrapper for a T4 (16GB).

Loads a 4-bit quantized instruct model and exposes judge_pair(), which returns
a verdict in {'A', 'B', 'TIE'} plus the raw output for logging.

Determinism: we use do_sample=False (greedy). This is what you want — you are
measuring verdict STABILITY under input transformations, so the judge itself
must not add sampling noise. Record everything anyway.

NOTE: this file is written for correctness but was NOT executed in a sandbox
with model access. Run the load cell in Colab first; if it errors, capture the
traceback and iterate.
"""

import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"

# The judge prompt. Kept minimal and explicit about output format so parsing
# is robust. Record prompt_version in every log row so you can change this
# without losing provenance.
PROMPT_VERSION = "v1"

_JUDGE_TEMPLATE = """You are an impartial evaluator. You will be shown a question and two candidate answers, labelled [Answer 1] and [Answer 2]. Decide which answer is better, judging only on correctness, relevance, and helpfulness. Ignore differences in length, formatting, and ordering.

Question:
{question}

[Answer 1]
{first}

[Answer 2]
{second}

Respond with EXACTLY one line in the form:
Verdict: 1
or
Verdict: 2
or
Verdict: tie"""


class OpenWeightJudge:
    def __init__(self, model_name: str = DEFAULT_MODEL, max_new_tokens: int = 16):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        self.model.eval()

    def _generate(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,          # greedy = deterministic
                temperature=None,
                top_p=None,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        gen = out[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(gen, skip_special_tokens=True)

    def judge_pair(self, question: str, first: str, second: str) -> dict:
        """
        Returns {'slot_verdict': '1'|'2'|'TIE', 'raw': str}.
        slot_verdict is in terms of SLOTS (1=first shown, 2=second shown).
        Caller maps slots back to A/B using first_slot_is_a from apply_position.
        """
        prompt = _JUDGE_TEMPLATE.format(question=question, first=first, second=second)
        raw = self._generate(prompt)
        slot = _parse_verdict(raw)
        return {"slot_verdict": slot, "raw": raw, "prompt_version": PROMPT_VERSION}


def _parse_verdict(raw: str) -> str:
    """
    Robust parse. Looks for 'Verdict: 1/2/tie' first; falls back to the first
    standalone 1/2/tie token. Returns 'UNPARSED' if nothing matches so you can
    audit failures rather than silently miscounting.
    """
    m = re.search(r"verdict\s*[:\-]?\s*(1|2|tie)", raw, re.IGNORECASE)
    if m:
        v = m.group(1).lower()
        return "TIE" if v == "tie" else v
    # fallback: first bare token
    m2 = re.search(r"\b(1|2|tie)\b", raw, re.IGNORECASE)
    if m2:
        v = m2.group(1).lower()
        return "TIE" if v == "tie" else v
    return "UNPARSED"


def slot_to_ab(slot_verdict: str, first_slot_is_a: bool) -> str:
    """Map a slot verdict (1/2/TIE) back to A/B/TIE using position bookkeeping."""
    if slot_verdict in ("TIE", "UNPARSED"):
        return slot_verdict
    if slot_verdict == "1":
        return "A" if first_slot_is_a else "B"
    else:  # '2'
        return "B" if first_slot_is_a else "A"
