"""Explicit writable output boundary for portable entry points."""
import os
from pathlib import Path

def check_output(path):
    value = os.environ.get('V7_RUNS_ROOT')
    if not value:
        raise ValueError('set V7_RUNS_ROOT to an absolute writable directory')
    root = Path(value).expanduser()
    if not root.is_absolute():
        raise ValueError('V7_RUNS_ROOT must be absolute')
    root = root.resolve()
    target = Path(path).resolve()
    if target == root or not target.is_relative_to(root):
        raise ValueError('use a named run directory inside V7_RUNS_ROOT')
    return target
