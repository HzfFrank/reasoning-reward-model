"""Optional Qwen3-VL visual patch embedding fast path.

The frozen visual stem is a Conv3d whose kernel and stride both equal the full
input patch. Every output therefore equals a dense projection of one flattened
patch. This opt-in replacement leaves weights, input ordering and dtype intact.
"""
from __future__ import annotations

import types
import torch.nn.functional as F


def enable_linear_patch_embedding(model) -> dict:
    stem = model.model.visual.patch_embed
    conv = stem.proj
    expected = (stem.temporal_patch_size, stem.patch_size, stem.patch_size)
    if tuple(conv.kernel_size) != expected or tuple(conv.stride) != expected:
        raise ValueError('visual patch convolution does not have full-patch kernel and stride')
    if tuple(conv.padding) != (0, 0, 0) or tuple(conv.dilation) != (1, 1, 1) or conv.groups != 1:
        raise ValueError('visual patch convolution is not equivalent to a flat linear projection')
    if conv.in_channels != stem.in_channels or conv.out_channels != stem.embed_dim:
        raise ValueError('visual patch projection shape mismatch')
    if hasattr(stem, '_original_forward'):
        raise ValueError('visual patch fast path already enabled')
    stem._original_forward = stem.forward

    def linear_forward(self, hidden_states):
        patches = hidden_states.reshape(
            -1, self.in_channels * self.temporal_patch_size * self.patch_size * self.patch_size
        ).to(dtype=self.proj.weight.dtype)
        weights = self.proj.weight.reshape(self.embed_dim, -1)
        return F.linear(patches, weights, self.proj.bias)

    stem.forward = types.MethodType(linear_forward, stem)
    return {'backend': 'linear_full_patch', 'kernel': expected,
            'in_features': conv.in_channels * expected[0] * expected[1] * expected[2],
            'out_features': conv.out_channels}


def disable_linear_patch_embedding(model) -> None:
    stem = model.model.visual.patch_embed
    if hasattr(stem, '_original_forward'):
        stem.forward = stem._original_forward
        del stem._original_forward
