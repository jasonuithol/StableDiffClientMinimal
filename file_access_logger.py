# file_access_logger.py
import builtins
import torch
import safetensors.torch as st
import safetensors.numpy as sn
from pathlib import Path

opened = set()

# --- Hook built-in open() ---
_orig_open = builtins.open
def logging_open(file, *args, **kwargs):
    try:
        opened.add(Path(file).resolve())
    except Exception:
        pass
    return _orig_open(file, *args, **kwargs)
builtins.open = logging_open

# --- Hook torch.load() ---
_orig_torch_load = torch.load
def logging_torch_load(f, *args, **kwargs):
    try:
        opened.add(Path(f).resolve())
    except Exception:
        pass
    return _orig_torch_load(f, *args, **kwargs)
torch.load = logging_torch_load

# --- Hook safetensors.torch.load_file() ---
_orig_st_load = st.load_file
def logging_st_load(path, *args, **kwargs):
    try:
        opened.add(Path(path).resolve())
    except Exception:
        pass
    return _orig_st_load(path, *args, **kwargs)
st.load_file = logging_st_load

# --- Hook safetensors.numpy.load_file() ---
_orig_sn_load = sn.load_file
def logging_sn_load(path, *args, **kwargs):
    try:
        opened.add(Path(path).resolve())
    except Exception:
        pass
    return _orig_sn_load(path, *args, **kwargs)
sn.load_file = logging_sn_load

# --- At-exit dump ---
def dump_manifest():
    print("\n=== FILES ACTUALLY ACCESSED ===")
    for f in sorted(opened):
        print(f)
import atexit
atexit.register(dump_manifest)