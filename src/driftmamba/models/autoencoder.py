"""Leakage-free dense autoencoder novelty scoring for flow features."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class FlowAutoencoder(nn.Module):
    def __init__(self, input_dimension: int, bottleneck: int = 8):
        super().__init__()
        hidden = max(16, input_dimension)
        self.network = nn.Sequential(
            nn.Linear(input_dimension, hidden), nn.GELU(), nn.Dropout(0.05),
            nn.Linear(hidden, bottleneck), nn.GELU(),
            nn.Linear(bottleneck, hidden), nn.GELU(), nn.Linear(hidden, input_dimension),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


@dataclass
class AutoencoderDetector:
    model: FlowAutoencoder
    error_center: float
    error_scale: float
    history: list[dict[str, float]]

    def knownness(self, features: np.ndarray, batch_size: int = 512) -> np.ndarray:
        errors = reconstruction_errors(self.model, features, batch_size)
        z = np.clip((errors - self.error_center) / self.error_scale, -60.0, 60.0)
        return 1.0 / (1.0 + np.exp(z))


def reconstruction_errors(model: FlowAutoencoder, features: np.ndarray,
                          batch_size: int = 512) -> np.ndarray:
    loader = DataLoader(TensorDataset(torch.from_numpy(features).float()), batch_size=batch_size)
    parts = []
    model.eval()
    with torch.no_grad():
        for (batch,) in loader:
            parts.append(torch.abs(batch - model(batch)).mean(dim=1).numpy())
    return np.concatenate(parts)


def train_autoencoder(train: np.ndarray, calibration: np.ndarray, *, epochs: int = 40,
                      batch_size: int = 256, seed: int = 42) -> AutoencoderDetector:
    torch.manual_seed(seed)
    model = FlowAutoencoder(train.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loader = DataLoader(TensorDataset(torch.from_numpy(train).float()), batch_size=batch_size,
                        shuffle=True)
    validation = torch.from_numpy(calibration).float()
    best_state, best_loss, stale, history = copy.deepcopy(model.state_dict()), float("inf"), 0, []
    for epoch in range(epochs):
        model.train()
        losses = []
        for (batch,) in loader:
            optimizer.zero_grad()
            loss = nn.functional.huber_loss(model(batch), batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.item()))
        model.eval()
        with torch.no_grad():
            validation_loss = float(nn.functional.huber_loss(model(validation), validation).item())
        history.append({"epoch": epoch + 1, "training_loss": float(np.mean(losses)),
                        "calibration_loss": validation_loss})
        if validation_loss < best_loss - 1e-6:
            best_loss, best_state, stale = validation_loss, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= 6:
                break
    model.load_state_dict(best_state)
    errors = reconstruction_errors(model, calibration)
    center = float(np.median(errors))
    scale = float(max(np.median(np.abs(errors - center)) * 1.4826, 1e-6))
    return AutoencoderDetector(model, center, scale, history)
