"""Environment shims copied verbatim from the as-run 03_track_b_evaluate.py (Windows paths, transformers 5.x, Jais remote code, bitsandbytes). Import before transformers models are loaded."""
import os, re as _re

# ── Redirect ALL HuggingFace/model downloads to F: drive ──────────────────────
os.environ.setdefault("HF_HOME", "F:/hf_cache")
os.environ.setdefault("TRANSFORMERS_CACHE", "F:/hf_cache/hub")
os.environ.setdefault("HF_DATASETS_CACHE", "F:/hf_cache/datasets")
# ──────────────────────────────────────────────────────────────────────────────

# ── Patch huggingface_hub validator to accept Windows absolute paths ───────────
# transformers 5.x calls validate_repo_id before checking if the path is local,
# so Windows paths like F:\... are rejected. Skip validation for absolute paths.
import huggingface_hub.utils._validators as _hfv
_orig_validate = _hfv.validate_repo_id
def _patched_validate_repo_id(repo_id, *args, **kwargs):
    if _re.match(r'^[A-Za-z]:[\\/]', str(repo_id)) or os.path.isabs(str(repo_id)):
        return  # local path — skip hub validation
    return _orig_validate(repo_id, *args, **kwargs)
_hfv.validate_repo_id = _patched_validate_repo_id
# also patch on the decorated wrapper used in file_download
import huggingface_hub.file_download as _hffd
if hasattr(_hffd, 'validate_repo_id'):
    _hffd.validate_repo_id = _patched_validate_repo_id
# ──────────────────────────────────────────────────────────────────────────────

import re
import json
import argparse
import torch
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

try:
    from bitsandbytes.nn import Params4bit as _Params4bit
    _orig_params4bit_new = _Params4bit.__new__
    def _patched_params4bit_new(cls, *args, **kwargs):
        kwargs.pop("_is_hf_initialized", None)
        return _orig_params4bit_new(cls, *args, **kwargs)
    _Params4bit.__new__ = staticmethod(_patched_params4bit_new)
except Exception:
    pass

# ── Transformers 5.x shim for Jais custom modeling_jais.py ────────────────────
from transformers.configuration_utils import PreTrainedConfig as _PTC
_orig_ga = _PTC.__getattribute__
def _patched_ga(self, name):
    if name == "add_cross_attention":
        try: return _orig_ga(self, name)
        except AttributeError: return False
    return _orig_ga(self, name)
_PTC.__getattribute__ = _patched_ga

from transformers.modeling_utils import PreTrainedModel as _PTM
_orig_initialize_missing_keys = _PTM._initialize_missing_keys
def _patched_initialize_missing_keys(self, is_quantized):
    if is_quantized:
        return
    return _orig_initialize_missing_keys(self, is_quantized)
_PTM._initialize_missing_keys = _patched_initialize_missing_keys
if not hasattr(_PTM, "get_head_mask"):
    def _get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked=False):
        if head_mask is None:
            return [None] * num_hidden_layers
        return head_mask
    _PTM.get_head_mask = _get_head_mask

import types as _t, sys as _s
_mpu = _t.ModuleType("transformers.utils.model_parallel_utils")
_mpu.assert_device_map = lambda device_map, num_blocks: None
_mpu.get_device_map = lambda num_blocks, device_ids: {i: device_ids[i % len(device_ids)] for i in range(num_blocks)}
_s.modules["transformers.utils.model_parallel_utils"] = _mpu
import transformers.utils as _tu; _tu.model_parallel_utils = _mpu

import transformers.pytorch_utils as _pt
if not hasattr(_pt, "find_pruneable_heads_and_indices"):
    def _fph(heads, n_heads, head_size, already_pruned):
        import torch as _torch
        mask = _torch.ones(n_heads, head_size)
        heads = set(heads) - already_pruned
        for h in sorted(heads):
            h -= sum(1 if p < h else 0 for p in already_pruned)
            mask[h] = 0
        mask = mask.view(-1).contiguous().eq(1)
        return heads, _torch.arange(len(mask))[mask].long()
    _pt.find_pruneable_heads_and_indices = _fph
if not hasattr(_pt, "prune_conv1d_layer"):
    def _pcl(layer, index, dim=1):
        import torch as _torch
        index = index.to(layer.weight.device)
        W = layer.weight.index_select(dim, index).clone().detach()
        b = layer.bias.clone().detach() if dim == 0 else layer.bias[index].clone().detach()
        sz = list(layer.weight.size()); sz[dim] = len(index)
        nl = _pt.Conv1D(sz[1], sz[0]).to(layer.weight.device)
        nl.weight.data.copy_(W.contiguous()); nl.bias.data.copy_(b.contiguous())
        return nl
    _pt.prune_conv1d_layer = _pcl
# ──────────────────────────────────────────────────────────────────────────────
