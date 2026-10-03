"""
Track B harness v2 — pure-Python part (no torch). Importable and unit-testable on CPU.

Replaces the as-run prompt builders / extract_answer() / get_gold_answer() of
original_as_run/03_track_b_evaluate.py. Every defect found by audit/audit_track_b.py is fixed here:

  D1  Gold lookup by guessing field names  -> explicit per-dataset adapters; a missing gold RAISES.
  D2  AraDiCE has no answer column         -> gold = Option A, as in the official lm-evaluation-harness
                                              task (lm_eval/tasks/aradice/cultural-benchmark: doc_to_target: 0),
                                              with a deterministic per-item shuffle so the gold position varies.
  D3  AraTrust options never shown          -> options rendered.
  D4  ArabCulture stem (first_statement) dropped, ArabicMMLU/MadinahQA Context dropped -> rendered.
  D5  Permissive extractor (any string starting with alef -> "A"; first alef/ba/jim/dal anywhere -> letter;
      otherwise returns text[:10] which is silently scored wrong)
                                           -> strict extractor with an explicit UNPARSED result (None),
                                              letters restricted to the options actually shown,
                                              and a degenerate-repetition detector.
  D6  Instruction always says "(A, B, C or D)" regardless of the number of options -> letters listed exactly.
"""
from __future__ import annotations

import ast
import hashlib
import random
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field

AR_LABELS = ["أ", "ب", "ج", "د", "هـ"]
EN_LABELS = ["A", "B", "C", "D", "E"]


@dataclass
class Item:
    item_id: str
    stem: str
    options: list[str]
    gold_index: int
    context: str = ""
    permutation: list[int] = field(default_factory=list)  # permutation[i] = original index shown at position i

    @property
    def gold_letter(self) -> str:
        return EN_LABELS[self.gold_index]


class GoldLabelError(ValueError):
    pass


def _blank(v) -> bool:
    return v is None or (isinstance(v, str) and v.strip() in ("", "None", "nan"))


def _as_dict(v):
    if isinstance(v, dict):
        return v
    if isinstance(v, str) and v.strip().startswith("{"):
        return ast.literal_eval(v)
    return None


def _letter_to_index(letter: str, n_options: int, where: str) -> int:
    s = str(letter).strip()
    table = {**{L: i for i, L in enumerate(EN_LABELS)}, **{L: i for i, L in enumerate(AR_LABELS)}, "ا": 0, "ه": 4}
    if s in table and table[s] < n_options:
        return table[s]
    if s.isdigit() and int(s) < n_options:  # 0-based index
        return int(s)
    raise GoldLabelError(f"{where}: cannot map gold {letter!r} to one of {n_options} options")


# ─────────────────────────────── dataset adapters ───────────────────────────────

def adapt_arabic_mmlu(ex: dict, idx: int) -> Item:
    opts = [str(ex[f"Option {k}"]).strip() for k in range(1, 6) if not _blank(ex.get(f"Option {k}"))]
    if _blank(ex.get("Answer Key")):
        raise GoldLabelError(f"arabic_mmlu[{idx}]: empty 'Answer Key'")
    return Item(str(ex.get("ID", idx)), str(ex.get("Question") or "").strip(), opts,
                _letter_to_index(ex["Answer Key"], len(opts), f"arabic_mmlu[{idx}]"),
                context="" if _blank(ex.get("Context")) else str(ex["Context"]).strip())


adapt_madinah_qa = adapt_arabic_mmlu  # identical schema (MBZUAI/MadinahQA)


def adapt_exams_ar(ex: dict, idx: int) -> Item:
    opts = [str(ex[k]).strip() for k in "ABCD" if not _blank(ex.get(k))]
    return Item(str(ex.get("id", idx)), str(ex["question"]).strip(), opts,
                _letter_to_index(ex["answer"], len(opts), f"exams_ar[{idx}]"))


_AR_PREFIX = re.compile(r"^\s*[\(\[]?\s*(أ|ب|ج|د|هـ|[A-E])\s*[\)\].:\-]\s*")


def adapt_aratrust(ex: dict, idx: int) -> Item:
    # AraTrust option strings carry their own prefix ("أ) ...", " ب) ..."); strip it, we re-label.
    opts = [_AR_PREFIX.sub("", str(ex[k])).strip() for k in "ABC" if not _blank(ex.get(k))]
    return Item(str(idx), str(ex["Question"]).strip(), opts,
                _letter_to_index(ex["Answer"], len(opts), f"aratrust[{idx}]"))


def adapt_arabculture(ex: dict, idx: int) -> Item:
    options = _as_dict(ex["options"])
    ak = _as_dict(ex["answer_key"])
    if not options or not ak:
        raise GoldLabelError(f"arabculture[{idx}]: options/answer_key not parseable")
    opts = [str(t).strip() for t in options["text"]]
    gold = ak.get("english_answer_key") or ak.get("arabic_answer_key")
    return Item(str(ex.get("sample_id", idx)), str(ex["first_statement"]).strip(), opts,
                _letter_to_index(gold, len(opts), f"arabculture[{idx}]"))


def adapt_aradice(ex: dict, idx: int) -> Item:
    # QCRI/AraDiCE-Culture ships no answer column. The official lm-evaluation-harness task
    # (lm_eval/tasks/aradice/cultural-benchmark/*.yaml) defines doc_to_target: 0, i.e. Option A is the
    # country-appropriate answer for every item. Positional bias is removed by shuffle_options() below.
    opts = [str(ex[f"Option {k}"]).strip() for k in "ABC"]
    return Item(str(ex["QID"]), str(ex["Question"]).strip(), opts, 0)


ADAPTERS = {
    "arabic_mmlu": adapt_arabic_mmlu,
    "madinah_qa": adapt_madinah_qa,
    "exams_ar": adapt_exams_ar,
    "aratrust": adapt_aratrust,
    "arabculture": adapt_arabculture,
    "aradice": adapt_aradice,
}
# Datasets whose gold position is constant by construction must be shuffled.
ALWAYS_SHUFFLE = {"aradice"}


def shuffle_options(item: Item, seed: int) -> Item:
    h = int(hashlib.sha256(f"{seed}:{item.item_id}".encode()).hexdigest(), 16)
    perm = list(range(len(item.options)))
    random.Random(h).shuffle(perm)
    return Item(item.item_id, item.stem, [item.options[j] for j in perm], perm.index(item.gold_index),
                item.context, perm)


def load_items(dataset_key: str, data, shuffle: bool = False, seed: int = 42) -> list[Item]:
    """Adapt every row; raise on the first unresolvable gold label (never silently score '')."""
    adapter = ADAPTERS[dataset_key]
    items = [adapter(dict(ex), i) for i, ex in enumerate(data)]
    if shuffle or dataset_key in ALWAYS_SHUFFLE:
        items = [shuffle_options(it, seed) for it in items]
    for it in items:
        if not it.options or not (0 <= it.gold_index < len(it.options)):
            raise GoldLabelError(f"{dataset_key}/{it.item_id}: invalid gold index {it.gold_index}")
    return items


# ─────────────────────────────── prompts ───────────────────────────────

def build_prompt(item: Item) -> str:
    labels = AR_LABELS[: len(item.options)]
    parts = []
    if item.context:
        parts.append(f"النص: {item.context}")
    parts.append(f"السؤال: {item.stem}" if item.stem else "السؤال:")
    parts.extend(f"{labels[i]}) {opt}" for i, opt in enumerate(item.options))
    allowed = "، ".join(labels[:-1]) + f"، أو {labels[-1]}" if len(labels) > 1 else labels[0]
    parts.append("")
    parts.append(f"اختر الإجابة الصحيحة ({allowed}). أجب بحرف واحد فقط.")
    parts.append("الإجابة:")
    return "\n".join(parts)


def option_labels(item: Item) -> list[str]:
    return AR_LABELS[: len(item.options)]


# ─────────────────────────────── strict extraction ───────────────────────────────

_DIACRITICS = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭـ]")  # harakat + tatweel
_LEAD = re.compile(
    r"^[\s\(\[«\"']*(?:(أ|ب|ج|د|هـ|[A-Ea-e])(?=$|[\s\)\].:\-،,؛»\"'])"   # printed labels
    r"|(ا|ه)(?=$|[\)\].:]))"                                              # bare alef/ha only with a closing mark
)
_LABELED = re.compile(
    r"(?:الإجابة|الاجابة|الجواب|الإجابة الصحيحة|answer)\s*(?:هي|is)?\s*[:：\-]?\s*[\(\[]?\s*(أ|ا|ب|ج|د|هـ|ه|[A-E])(?=$|[\s\)\].:\-،,])",
    re.IGNORECASE,
)
_MAP = {"أ": 0, "ا": 0, "ب": 1, "ج": 2, "د": 3, "هـ": 4, "ه": 4, **{L: i for i, L in enumerate(EN_LABELS)}}


def normalize(text: str) -> str:
    return _DIACRITICS.sub("", unicodedata.normalize("NFKC", text or "")).strip()


def is_degenerate(text: str, min_len: int = 8, share: float = 0.6) -> bool:
    """True for outputs like 'ألللللللللللللل' or 'َََََََََ' (one character dominating)."""
    body = re.sub(r"\s", "", text or "")
    if len(body) < min_len:
        return False
    return Counter(body).most_common(1)[0][1] / len(body) >= share


def extract_choice(generated: str, item: Item) -> tuple[int | None, str]:
    """Return (option index or None, rule). None means UNPARSED — it is reported, not guessed."""
    raw = generated or ""
    if not raw.strip():
        return None, "empty"
    if is_degenerate(raw):
        return None, "degenerate"
    t = normalize(raw)
    n = len(item.options)
    m = _LEAD.match(t)
    if m:
        g = m.group(1) or m.group(2)
        i = _MAP[g.upper() if g.isascii() else g]
        if i < n:
            return i, "leading_label"
    m = _LABELED.search(t)
    if m:
        i = _MAP[m.group(1).upper() if m.group(1).isascii() else m.group(1)]
        if i < n:
            return i, "labeled_answer"
    # answer given as option text
    tn = t.split("\n")[0]
    hits = [i for i, o in enumerate(item.options) if normalize(o) and (tn.startswith(normalize(o)) or normalize(o) == tn)]
    if len(hits) == 1:
        return hits[0], "option_text"
    return None, "no_rule"


# ─────────────────────────────── reporting helpers ───────────────────────────────

def baselines(items: list[Item]) -> dict:
    n = len(items)
    counts = Counter(it.gold_letter for it in items)
    return {
        "chance": sum(1 / len(it.options) for it in items) / n,
        "constant_letter_acc": {L: counts.get(L, 0) / n for L in EN_LABELS if counts.get(L, 0)},
        "best_constant_acc": max(counts.values()) / n,
    }


def summarize(items: list[Item], preds: list[int | None], rules: list[str]) -> dict:
    n = len(items)
    correct = sum(1 for it, p in zip(items, preds) if p is not None and p == it.gold_index)
    parsed = [p for p in preds if p is not None]
    dist = Counter(EN_LABELS[p] for p in parsed)
    b = baselines(items)
    return {
        "n": n,
        "correct": correct,
        "accuracy": correct / n,                       # unparsed counted as wrong
        "parse_rate": len(parsed) / n,
        "accuracy_on_parsed": (correct / len(parsed)) if parsed else None,
        "prediction_distribution": dict(dist),
        "max_prediction_share": (max(dist.values()) / len(parsed)) if parsed else None,
        "rule_counts": dict(Counter(rules)),
        "degenerate_rate": rules.count("degenerate") / n,
        **b,
        "above_best_constant": (correct / n) > b["best_constant_acc"],
    }
