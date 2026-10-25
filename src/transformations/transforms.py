"""
Surface transformations for the factorial judge-bias study.

Three factors, each with a control (0) and perturbed (1) level:
    position  : 0 = A first,  1 = A second (order swapped)
    formatting: 0 = plain,    1 = structured/markdown
    length    : 0 = original, 1 = neutral-filler expansion

CRITICAL DESIGN RULE (see paper section 4):
    Formatting and length must NOT change substantive content.
    - Formatting only re-wraps existing sentences as markdown structure.
    - Length only adds semantically null connective/restating filler.
    If you cannot guarantee this for `length`, drop the length factor and
    run the 2x2 position x formatting design instead. A clean 2x2 beats a
    contaminated 2x3.
"""

import re


# ----------------------------------------------------------------------
# POSITION  (operates on the PAIR, not a single response)
# ----------------------------------------------------------------------
def apply_position(resp_a: str, resp_b: str, level: int):
    """
    level 0: (A, B) in slots (first, second)
    level 1: (B, A) — order swapped

    Returns (first_slot_text, second_slot_text, first_slot_is_a).
    `first_slot_is_a` lets you map the judge's slot verdict back to A/B.
    """
    if level == 0:
        return resp_a, resp_b, True
    else:
        return resp_b, resp_a, False


# ----------------------------------------------------------------------
# FORMATTING  (single response; content-preserving)
# ----------------------------------------------------------------------
def apply_formatting(text: str, level: int) -> str:
    """
    level 0: plain text, unchanged.
    level 1: same sentences re-presented as a markdown list under a heading.

    We only re-wrap existing sentences. No words are added or removed beyond
    list markers and a generic heading, so substantive content is fixed.
    """
    if level == 0:
        return text

    # Split into sentences without dropping content. Keep it simple and
    # deterministic; over-engineering the splitter risks altering text.
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return text
    bullets = "\n".join(f"- {s}" for s in sentences)
    return f"### Explanation\n{bullets}"


# ----------------------------------------------------------------------
# LENGTH  (single response; the DANGEROUS factor — content-preserving filler)
# ----------------------------------------------------------------------
# Neutral connectives / restatements that add tokens WITHOUT adding claims.
# Keep these strictly non-substantive. Validate a sample by hand (see
# validate_length_preservation below) before trusting results.
_FILLER_OPENERS = [
    "To put it another way, ",
    "In other words, ",
    "Stated differently, ",
    "As noted, ",
]
_FILLER_CLOSERS = [
    " This point is worth keeping in mind.",
    " That is the key idea here.",
    " This is the essential reasoning.",
]


def apply_length(text: str, level: int) -> str:
    """
    level 0: original.
    level 1: expanded with neutral filler — restates existing sentences with
             connectives, adds no new factual propositions.

    Method: for each original sentence, optionally prepend a neutral opener
    (a restating connective) and append a neutral closer. This roughly
    doubles token count while introducing zero new content nouns.

    WARNING: verify with validate_length_preservation() on a sample. If the
    check flags added content, fall back to the 2x2 design.
    """
    if level == 0:
        return text

    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return text

    out = []
    for i, s in enumerate(sentences):
        opener = _FILLER_OPENERS[i % len(_FILLER_OPENERS)]
        closer = _FILLER_CLOSERS[i % len(_FILLER_CLOSERS)]
        # Lowercase the first letter of s after an opener for readability;
        # purely cosmetic, adds no content.
        s_low = s[0].lower() + s[1:] if s else s
        out.append(f"{s}{closer} {opener}{s_low}")
    return " ".join(out)


# ----------------------------------------------------------------------
# VALIDATION  (run this on a sample before trusting the length factor)
# ----------------------------------------------------------------------
_CONTENT_WORD_RE = re.compile(r"\b[a-z]{4,}\b")
_FILLER_WORDS = set()
for _phrase in _FILLER_OPENERS + _FILLER_CLOSERS:
    for _w in re.findall(r"[a-z]{4,}", _phrase.lower()):
        _FILLER_WORDS.add(_w)


def validate_length_preservation(original: str, expanded: str) -> dict:
    """
    Cheap rule-based check that `expanded` introduced no new content words
    beyond the known filler vocabulary. Returns a dict you can log and
    report in an appendix.

    `new_content_words` should be EMPTY if the expansion is truly neutral.
    If it is not empty, inspect manually — your filler may be leaking
    content, or the splitter mangled the text.
    """
    orig_words = set(_CONTENT_WORD_RE.findall(original.lower()))
    exp_words = set(_CONTENT_WORD_RE.findall(expanded.lower()))
    added = exp_words - orig_words - _FILLER_WORDS
    return {
        "orig_len_chars": len(original),
        "exp_len_chars": len(expanded),
        "length_ratio": round(len(expanded) / max(len(original), 1), 2),
        "new_content_words": sorted(added),
        "clean": len(added) == 0,
    }
