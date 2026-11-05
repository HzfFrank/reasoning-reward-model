"""Shared fixed storyboard sampling and paired-observation checks."""
from __future__ import annotations

def uniform_storyboard_frames(frames: list[int], count: int = 8) -> list[int]:
    ordered=sorted(set(frames))
    if len(ordered)<count:
        raise ValueError(f'need at least {count} paired time points, got {len(ordered)}')
    selected=[ordered[round(i*(len(ordered)-1)/(count-1))] for i in range(count)]
    if len(set(selected)) != count:
        raise ValueError(f'uniform sampling produced duplicate storyboard frames: {selected}')
    return selected
