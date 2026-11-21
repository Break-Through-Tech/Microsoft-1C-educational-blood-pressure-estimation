"""Wrapper for Claire's pretrained 1D ResNet (resnet_bp_v3.pt). Inference only, CPU, never retrained.

The architecture below reproduces model_training.ipynb (cells "1D ResNet") so the saved state_dict
loads with strict=True; tests/test_resnet.py checks the outputs match the notebook's own class.
Checkpoint: dict with 'model' (state_dict), 'y_mean', 'y_std' (mmHg, training split), 'config', 'history'.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ..preprocess import TARGET_FS, WINDOW_SIZE
from .base import BPModel, check_output

N_SEGMENTS = 10
BATCH_SIZE = 1024


class ResidualBlock(nn.Module):
    """Two conv-BN-ReLU layers with a skip connection, then max-pool /2."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dropout: float = 0.0):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.skip = (nn.Identity() if in_ch == out_ch else
                     nn.Sequential(nn.Conv1d(in_ch, out_ch, 1, bias=False), nn.BatchNorm1d(out_ch)))
        self.pool = nn.MaxPool1d(2)
        self.drop = nn.Dropout1d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out + self.skip(x))
        return self.drop(self.pool(out))


class PPGResNet(nn.Module):
    def __init__(self, input_len: int = WINDOW_SIZE, in_channels: int = 1, filters=(32, 64, 128),
                 kernel_size: int = 3, use_gap: bool = False, conv_dropout: float = 0.0, n_outputs: int = 2):
        super().__init__()
        blocks, in_ch = [], in_channels
        for f in filters:
            blocks.append(ResidualBlock(in_ch, f, kernel_size, conv_dropout))
            in_ch = f
        self.features = nn.Sequential(*blocks)
        if use_gap:
            flatten, n_flat = [nn.AdaptiveAvgPool1d(1), nn.Flatten()], in_ch
        else:
            flatten, n_flat = [nn.Flatten()], in_ch * (input_len // 2 ** len(filters))
        self.head = nn.Sequential(*flatten, nn.Linear(n_flat, 512), nn.ReLU(inplace=True),
                                  nn.Dropout(0.5), nn.Linear(512, n_outputs))

    def forward(self, x):
        return self.head(self.features(x))


# checkpoints saved before the config existed are the notebook's v1 baseline
BASELINE_CONFIG = {"use_gap": False, "kernel": 3, "input_channels": ["ppg"]}


class ResNetBP(BPModel):
    name = "resnet_v3"
    display_name = "1D ResNet (v3, PPG)"
    input_kind = "waveform"

    def __init__(self, net: PPGResNet, y_mean: np.ndarray, y_std: np.ndarray, config: dict, source: str = ""):
        self.net = net.eval()
        self.y_mean = np.asarray(y_mean, dtype=np.float32)
        self.y_std = np.asarray(y_std, dtype=np.float32)
        self.config = config
        self.source = source

    @classmethod
    def load(cls, path: Path) -> "ResNetBP":
        # weights_only=False: the checkpoint pickles numpy arrays (y_mean / y_std). Team-trusted files only.
        ckpt = torch.load(Path(path), map_location="cpu", weights_only=False)
        cfg = ckpt.get("config", BASELINE_CONFIG)
        if list(cfg["input_channels"]) != ["ppg"]:
            raise ValueError(f"{path} expects inputs {cfg['input_channels']}; the app's standard input is PPG only")
        net = PPGResNet(in_channels=1, kernel_size=cfg["kernel"], use_gap=cfg["use_gap"],
                        conv_dropout=cfg.get("conv_dropout", 0.0))
        net.load_state_dict(ckpt["model"], strict=True)
        return cls(net, ckpt["y_mean"], ckpt["y_std"], cfg, source=str(path))

    @staticmethod
    def _as_tensor(X: np.ndarray) -> torch.Tensor:
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 2:
            X = X[:, None, :]
        if X.ndim != 3 or X.shape[1:] != (1, WINDOW_SIZE):
            raise ValueError(f"expected windows of shape (n, 1, {WINDOW_SIZE}), got {X.shape}")
        return torch.from_numpy(np.ascontiguousarray(X))

    def predict(self, X: np.ndarray) -> np.ndarray:
        n = len(X)
        out = []
        self.net.eval()
        with torch.inference_mode():
            for s in range(0, n, BATCH_SIZE):
                out.append(self.net(self._as_tensor(X[s:s + BATCH_SIZE])).numpy())
        pred = (np.concatenate(out) if out else np.zeros((0, 2))) * self.y_std + self.y_mean
        return check_output(pred, n)

    def explain(self, X: np.ndarray, n_segments: int = N_SEGMENTS) -> dict:
        """Gradient saliency |d prediction / d input| per target, averaged into n_segments time segments.

        Shares are normalized per window and target to sum to 1. Saliency shows where the model is
        sensitive, not why; treat it as a teaching aid. Gradients need autograd, so this runs under
        enable_grad (eval mode, so dropout and batch norm behave as at inference).
        """
        x = self._as_tensor(X).clone().requires_grad_(True)
        self.net.eval()
        shares = {}
        with torch.enable_grad():
            out = self.net(x)                       # windows are independent in eval mode
            for k, target in enumerate(("sbp", "dbp")):
                grad, = torch.autograd.grad(out[:, k].sum(), x, retain_graph=(k == 0))
                sal = grad.abs()[:, 0, :].numpy()                                        # (n, 625)
                seg = np.stack([c.mean(axis=1) for c in np.array_split(sal, n_segments, axis=1)], axis=1)
                total = seg.sum(axis=1, keepdims=True)
                shares[target] = np.divide(seg, total, out=np.zeros_like(seg), where=total > 0)
        bounds = np.array_split(np.arange(WINDOW_SIZE), n_segments)
        return {
            "method": "gradient_saliency",
            "description": "share of |gradient| of each prediction w.r.t. the input PPG, per time segment",
            "segments": [{"start_s": round(b[0] / TARGET_FS, 3), "end_s": round((b[-1] + 1) / TARGET_FS, 3)}
                         for b in bounds],
            "sbp": shares["sbp"].round(4).tolist(),
            "dbp": shares["dbp"].round(4).tolist(),
        }

    def info(self) -> dict:
        return {**super().info(), "source": self.source, "config": self.config,
                "trained_by": "Claire (origin/claire, notebooks/model_training.ipynb)", "retrained": False}
