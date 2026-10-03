"""
One-time utility: convert Jais fp32 safetensors weights → fp16 and save to
F:/hf_cache/jais-family-13b-chat-fp16/

Run once before 03_track_b_evaluate.py:
    ./venv/Scripts/python convert_jais_fp16.py
"""

import os, json, shutil, glob, tempfile
import torch
from safetensors.torch import load_file, save_file

# Use F: drive for all temp/cache operations — C: is full
os.environ.setdefault("HF_HOME", "F:/hf_cache")
os.environ.setdefault("TRANSFORMERS_CACHE", "F:/hf_cache/hub")
os.environ.setdefault("TEMP", "F:/tmp")
os.environ.setdefault("TMP", "F:/tmp")
os.makedirs("F:/tmp", exist_ok=True)
tempfile.tempdir = "F:/tmp"

SRC_DIR = "F:/hf_cache/hub/models--inceptionai--jais-family-13b-chat/snapshots/0ef8b4f80429609890816d912b331d3b95864707"
DST_DIR = "F:/hf_cache/jais-family-13b-chat-fp16"

os.makedirs(DST_DIR, exist_ok=True)

# Copy all non-weight files as-is
for f in os.listdir(SRC_DIR):
    if not f.endswith(".safetensors"):
        src = os.path.join(SRC_DIR, f)
        dst = os.path.join(DST_DIR, f)
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
            print(f"  Copied: {f}")

# Convert each shard
shards = sorted(glob.glob(os.path.join(SRC_DIR, "*.safetensors")))
print(f"\nConverting {len(shards)} shards fp32 -> fp16...")

for shard_path in shards:
    fname = os.path.basename(shard_path)
    dst_path = os.path.join(DST_DIR, fname)

    if os.path.exists(dst_path):
        print(f"  Skipping (exists): {fname}")
        continue

    print(f"  Converting: {fname}  ", end="", flush=True)
    tensors = load_file(shard_path, device="cpu")
    fp16_tensors = {k: v.to(torch.float16) for k, v in tensors.items()}
    save_file(fp16_tensors, dst_path)
    del tensors, fp16_tensors
    import gc; gc.collect()
    size_gb = os.path.getsize(dst_path) / 1024**3
    print(f"-> {size_gb:.1f} GB")

print(f"\nDone. fp16 weights saved to: {DST_DIR}")
print("Now update 03_track_b_evaluate.py jais path to:", DST_DIR)
