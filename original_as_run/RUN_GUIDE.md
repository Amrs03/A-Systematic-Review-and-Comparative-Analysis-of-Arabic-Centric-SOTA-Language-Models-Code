> [!NOTE]
> **Provenance — the run guide as it was used, preserved verbatim.**
>
> These are the commands that produced the artifacts in `original_results/`. They are kept
> for the record only. The defects documented in [`AUDIT_REPORT.md`](../AUDIT_REPORT.md)
> are present in the scripts this guide drives; use [`pipeline/`](../pipeline) instead.

# Arabic NLP Leaderboard — Run Guide
## RTX 4060 (8GB VRAM) + 64GB RAM

---

## Install Dependencies (run once)

```bash
pip install torch transformers datasets evaluate accelerate peft bitsandbytes seqeval scikit-learn tqdm
```

---

## Execution Order — Copy-Paste These Commands

### Step 1: Download & prepare all datasets (~10 min)
```bash
python 01_setup_datasets.py
```
✅ **Check:** Look at the output — all datasets should show ✅. Fix any ❌ before continuing.

⚠️ **Common issues:**
- Some HuggingFace dataset IDs may have changed — check the HF page manually
- AOC may need manual GitHub download
- QADI may be under a different org name

---

### Step 2: Track A — Fine-tune AraBERT & MARBERT (~3-5 hours total)
```bash
# Run everything in one go (recommended — walk away and let it run):
python 02_track_a_finetune.py --all

# OR run individually if you want to monitor:
python 02_track_a_finetune.py --model arabert --dataset hard
python 02_track_a_finetune.py --model arabert --dataset astd
python 02_track_a_finetune.py --model arabert --dataset arsarcasm
python 02_track_a_finetune.py --model arabert --dataset qadi
python 02_track_a_finetune.py --model arabert --dataset anercorp
python 02_track_a_finetune.py --model marbert --dataset hard
python 02_track_a_finetune.py --model marbert --dataset astd
python 02_track_a_finetune.py --model marbert --dataset arsarcasm
python 02_track_a_finetune.py --model marbert --dataset qadi
python 02_track_a_finetune.py --model marbert --dataset anercorp

# If a run crashes, resume without re-running completed ones:
python 02_track_a_finetune.py --all --skip-existing
```

✅ **Check:** `results/track_a/` should have 10 JSON files (we dropped AOC since it needs GitHub download — add it back if you got it).

---

### Step 3: Track B — Evaluate LLMs (~4-6 hours total)
```bash
# Run everything (loads each model once, runs all datasets, then unloads):
python 03_track_b_evaluate.py --all

# OR one at a time:
python 03_track_b_evaluate.py --model allam --dataset arabic_mmlu
# ... etc

# Resume:
python 03_track_b_evaluate.py --all --skip-existing
```

⚠️ **Jais-13B will be SLOW** because it uses CPU offloading. Budget ~2 hours for Jais alone.
If it's too slow, skip Jais and note "N/A" in the leaderboard.

✅ **Check:** `results/track_b/` should have up to 18 JSON files.

---

### Step 4: Bridge Tasks (~1-2 hours)
```bash
python 04_bridge_evaluate.py --all
```

✅ **Check:** `results/bridge/` should have 6 JSON files.

---

### Step 5: Build the Leaderboard (~instant)
```bash
python 05_build_leaderboard.py
```

This prints the full leaderboard to terminal AND exports:
- `results/leaderboard.csv` — for spreadsheets
- `results/leaderboard_full.json` — for programmatic use

---

## ⏱️ Realistic Time Estimates for RTX 4060

| Step | Task | Time |
|------|------|------|
| 1 | Dataset download | 10-15 min |
| 2 | Track A (10 fine-tuning runs) | 3-5 hours |
| 3 | Track B (18 zero-shot evals) | 4-6 hours |
| 4 | Bridge (6 evals) | 1-2 hours |
| 5 | Build leaderboard | instant |
| **Total** | | **8-13 hours** |

**Strategy:** Start Step 2 tonight, let it run overnight. Do Steps 3-5 tomorrow.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| CUDA out of memory (Track A) | Reduce `per_device_train_batch_size` to 4 in the script |
| CUDA out of memory (Track B) | Already using 4-bit; reduce `max_length` to 512 |
| Jais too slow | Skip it, add scores from papers instead |
| Dataset won't load | Check HuggingFace page for correct dataset ID |
| Model won't download | Check your internet, try `huggingface-cli login` |
| Labels don't match | Check `01_setup_datasets.py` output for column names, update config |
| Low scores | Check prompt template, increase MAX_LENGTH, try few-shot |
