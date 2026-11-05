"""Opt-in final episode inference using the validated visual patch backend.

All query construction, generation, parser, stride, feedback and output logic
remain in final_episode_inference.py. This file does not affect existing runs.
"""
from __future__ import annotations

import final_episode_inference as original
from visual_patch_linear import enable_linear_patch_embedding


_load = original.load_checked_model


def _load_with_linear_patch(export, device):
    model, manifest = _load(export, device)
    backend = enable_linear_patch_embedding(model)
    manifest['visual_patch_backend'] = backend
    return model, manifest


original.load_checked_model = _load_with_linear_patch


if __name__ == '__main__':
    original.main()
