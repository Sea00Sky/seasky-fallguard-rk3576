"""ST-GCN++ export wrapper."""
from __future__ import annotations

import torch
import torch.nn as nn


def _patched_unit_gcn_forward(self, x, A=None):
    """`unit_gcn.forward` with the einsum replaced by matmul + reduce."""
    n, c, t, v = x.shape
    res = self.down(x) if self.with_res else 0

    A_switch = {None: self.A, "init": self.A}
    if hasattr(self, "PA"):
        A_switch.update({"offset": self.A + self.PA, "importance": self.A * self.PA})
    A = A_switch[self.adaptive]

    if self.conv_pos == "pre":
        x = self.conv(x)                              # (n, K*C, t, v)
        x = x.view(n, self.num_subsets, -1, t, v)     # (n, K, C, t, v)
        # einsum('nkctv,kvw->nctw'): contract v, keep w as the new joint axis.
        # Flatten (c, t) into one axis so the matmul batch dims are (n, k) vs (k).
        x = x.reshape(n, self.num_subsets, -1, v)     # (n, K, C*t, v)
        x = torch.matmul(x, A)                        # (n, K, C*t, w)
        x = x.sum(dim=1).view(n, -1, t, A.shape[2])   # (n, C, t, w)
    elif self.conv_pos == "post":
        # einsum('nctv,kvw->nkctw'): no contraction, w is appended.
        x = torch.matmul(x.unsqueeze(1), A)           # (n, K, c, t, w)
        x = x.reshape(n, -1, t, A.shape[2])           # (n, K*c, t, w)
        x = self.conv(x)
    else:  # pragma: no cover - guarded by unit_gcn's own assert
        raise ValueError(f"unsupported conv_pos: {self.conv_pos}")

    return self.act(self.bn(x) + res)


def patch_einsum(model: nn.Module) -> int:
    """Replace `unit_gcn.forward` on every block. Returns how many were patched."""
    from pyskl.models.gcns.utils.gcn import unit_gcn

    n = 0
    for module in model.modules():
        if isinstance(module, unit_gcn):
            module.forward = _patched_unit_gcn_forward.__get__(module, unit_gcn)
            n += 1
    return n


class STGCNExport(nn.Module):
    """`RecognizerGCN` -> a module whose `forward` returns (N, 2) logits."""

    def __init__(self, recognizer: nn.Module, patch: bool = True) -> None:
        super().__init__()
        self.recognizer = recognizer
        self.n_patched = patch_einsum(self.recognizer) if patch else 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, M, T, V, C) float32 -> logits (N, num_classes)."""
        feat = self.recognizer.backbone(x)      # (N, M, C', T', V)
        logits = self.recognizer.cls_head(feat)  # (N, num_classes)
        return logits


class STGCNLogits(nn.Module):
    """Unpatched reference: the same two calls, einsum left alone."""

    def __init__(self, recognizer: nn.Module) -> None:
        super().__init__()
        self.recognizer = recognizer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.recognizer.cls_head(self.recognizer.backbone(x))
