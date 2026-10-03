"""
CPU-only regression tests for harness.py, run against the real prepared_data.

    PREPARED_DATA=/path/to/ARABIC/results/prepared_data python -m pytest -q tests/

These are the checks that would have caught every Track B defect before any GPU hour was spent.
"""
import os
import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import harness as H  # noqa: E402

DATA = Path(os.environ.get("PREPARED_DATA", "prepared_data"))
DATASETS = list(H.ADAPTERS)
needs_data = pytest.mark.skipif(not DATA.exists(), reason="set PREPARED_DATA")


def _items(key, **kw):
    from datasets import load_from_disk
    return H.load_items(key, load_from_disk(str(DATA / key))["test"], **kw)


# ── gold labels ────────────────────────────────────────────────────────────────
@needs_data
@pytest.mark.parametrize("key", DATASETS)
def test_every_item_has_a_resolvable_gold(key):
    items = _items(key)
    assert items and all(0 <= it.gold_index < len(it.options) for it in items)


@needs_data
@pytest.mark.parametrize("key", DATASETS)
def test_constant_predictor_scores_exactly_its_baseline_and_nonzero(key):
    """Sanity gate: an always-'A' model must score share(gold==A) — never a structural 0."""
    items = _items(key)
    preds = [0] * len(items)
    s = H.summarize(items, preds, ["leading_label"] * len(items))
    assert s["accuracy"] == pytest.approx(s["constant_letter_acc"].get("A", 0.0))
    assert s["accuracy"] > 0


@needs_data
def test_aradice_gold_position_is_not_constant_after_shuffle():
    c = Counter(it.gold_index for it in _items("aradice"))
    assert len(c) == 3 and max(c.values()) / sum(c.values()) < 0.5


# ── prompts ────────────────────────────────────────────────────────────────────
@needs_data
@pytest.mark.parametrize("key", DATASETS)
def test_prompt_shows_every_option_and_a_stem(key):
    for it in _items(key)[:300]:
        p = H.build_prompt(it)
        for lab, opt in zip(H.option_labels(it), it.options):
            assert f"{lab}) {opt}" in p
        assert it.stem or it.context, f"{key}/{it.item_id} has neither stem nor context"


@needs_data
def test_arabculture_stem_is_first_statement():
    it = _items("arabculture")[0]
    assert it.stem.startswith("تحضر أم ماجد الفلافل") and it.stem in H.build_prompt(it)


@needs_data
def test_madinah_context_is_rendered():
    items = [it for it in _items("madinah_qa") if it.context]
    assert len(items) >= 600
    assert all(it.context[:40] in H.build_prompt(it) for it in items)


@needs_data
def test_aratrust_options_rendered_without_duplicate_prefix():
    it = _items("aratrust")[0]
    p = H.build_prompt(it)
    assert "أ) أنثى الحيوان" in p and "أ) أ)" not in p and it.gold_letter == "B"


# ── extractor ──────────────────────────────────────────────────────────────────
ITEM4 = H.Item("x", "q", ["o1", "o2", "o3", "o4"], 1)
ITEM3 = H.Item("y", "q", ["شاي", "قهوة", "ماء"], 2)


@pytest.mark.parametrize("raw,expected,rule", [
    (" ألللللللللللللللللل", None, "degenerate"),      # the actual Falcon output
    (" َََََََََ", None, "degenerate"),                # actual Falcon EXAMS output (fatha run)
    (" علللللللللللللللللل", None, "degenerate"),      # actual Falcon AraDiCE output
    ("", None, "empty"),
    ("ب", 1, "leading_label"),
    ("ب) o2", 1, "leading_label"),
    (" (ج) لأن ...", 2, "leading_label"),
    ("B.", 1, "leading_label"),
    ("الإجابة الصحيحة هي: د", 3, "labeled_answer"),
    ("بناءً على المعطيات فإن", None, "no_rule"),        # word starting with ب is NOT an answer
    ("اعتقد ان", None, "no_rule"),                      # word starting with alef is NOT 'A'
    ("هـ", None, "no_rule"),                            # E is not among 4 options
])
def test_extractor(raw, expected, rule):
    assert H.extract_choice(raw, ITEM4) == (expected, rule)


def test_extractor_option_text():
    assert H.extract_choice("ماء", ITEM3) == (2, "option_text")


def test_as_run_extractor_false_positive_is_documented():
    """The as-run rule `text.startswith('أ') -> 'A'` turns garbage into a valid prediction; v2 must not."""
    assert H.extract_choice(" ألللل ألللل ألللل", ITEM4)[0] is None


@pytest.mark.parametrize("raw,expected", [("ا لس نزل", None), ("ا)", 0), ("ا.", 0), ("ا", 0), ("ه الجواب", None)])
def test_bare_alef_needs_closing_mark(raw, expected):
    assert H.extract_choice(raw, H.Item("z", "q", ["a", "b", "c", "d", "e"], 0))[0] == expected
