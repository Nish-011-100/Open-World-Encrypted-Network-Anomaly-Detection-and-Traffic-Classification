"""Training and batched inference for the multi-view prototype model."""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from driftmamba.models.driftmamba import (
    DriftMambaClassifier,
    FlowTransformerClassifier,
    HyenaFlowClassifier,
    XLSTMFlowClassifier,
    prototype_loss,
)


@dataclass
class DeepTrainingResult:
    model: DriftMambaClassifier | FlowTransformerClassifier | HyenaFlowClassifier | XLSTMFlowClassifier
    label_encoder: LabelEncoder
    history: list[dict[str, float]]
    configuration: dict[str, int]


def load_tensors(path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=False)
    return {name: data[name] for name in data.files}


def _dataset(data: dict[str, np.ndarray], labels: np.ndarray) -> TensorDataset:
    return TensorDataset(
        torch.from_numpy(data["sequence"]).float(),
        torch.from_numpy(data["mask"]).bool(),
        torch.from_numpy(data["aggregate"]).float(),
        torch.from_numpy(data["signatures"]).float(),
        torch.from_numpy(labels).long(),
    )


def train_deep_model(train: dict[str, np.ndarray], calibration: dict[str, np.ndarray], *,
                     epochs: int = 30, batch_size: int = 256, patience: int = 6,
                     model_dimension: int = 64, embedding_dimension: int = 64,
                     blocks: int = 3, prototypes_per_class: int = 2,
                     balanced_sampling: bool = False,
                     encoder_type: str = "driftmamba",
                     seed: int = 42) -> DeepTrainingResult:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    encoder = LabelEncoder().fit(train["labels"].astype(str))
    train_targets = encoder.transform(train["labels"].astype(str))
    calibration_targets = encoder.transform(calibration["labels"].astype(str))
    if balanced_sampling:
        class_counts = np.bincount(train_targets, minlength=len(encoder.classes_))
        sample_weights = 1.0 / np.maximum(class_counts[train_targets], 1)
        sampler = WeightedRandomSampler(
            torch.as_tensor(sample_weights, dtype=torch.double), len(sample_weights), replacement=True
        )
        train_loader = DataLoader(_dataset(train, train_targets), batch_size=batch_size, sampler=sampler)
    else:
        train_loader = DataLoader(_dataset(train, train_targets), batch_size=batch_size, shuffle=True)
    calibration_loader = DataLoader(
        _dataset(calibration, calibration_targets), batch_size=batch_size, shuffle=False
    )
    configuration = {
        "aggregate_dimension": int(train["aggregate"].shape[1]),
        "number_classes": len(encoder.classes_), "model_dimension": model_dimension,
        "embedding_dimension": embedding_dimension, "blocks": blocks,
        "prototypes_per_class": prototypes_per_class,
    }
    if encoder_type == "driftmamba":
        model = DriftMambaClassifier(**configuration)
    elif encoder_type == "transformer":
        model = FlowTransformerClassifier(**configuration)
    elif encoder_type == "hyena":
        model = HyenaFlowClassifier(**configuration)
    elif encoder_type == "xlstm":
        model = XLSTMFlowClassifier(**configuration)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_state = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    stale_epochs = 0
    history = []
    for epoch in range(epochs):
        model.train()
        training_losses = []
        for sequence, mask, aggregate, signatures, targets in train_loader:
            optimizer.zero_grad()
            logits, _, similarities = model(sequence, mask, aggregate, signatures)
            loss = prototype_loss(logits, similarities, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            training_losses.append(loss.item())
        model.eval()
        validation_losses = []
        with torch.no_grad():
            for sequence, mask, aggregate, signatures, targets in calibration_loader:
                logits, _, similarities = model(sequence, mask, aggregate, signatures)
                validation_losses.append(prototype_loss(logits, similarities, targets).item())
        validation_loss = float(np.mean(validation_losses))
        history.append({"epoch": epoch + 1, "training_loss": float(np.mean(training_losses)),
                        "calibration_loss": validation_loss})
        if validation_loss < best_loss - 1e-5:
            best_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break
    model.load_state_dict(best_state)
    return DeepTrainingResult(model, encoder, history, configuration)


def predict_deep(model: DriftMambaClassifier | FlowTransformerClassifier | HyenaFlowClassifier |
                 XLSTMFlowClassifier,
                 data: dict[str, np.ndarray],
                 batch_size: int = 512) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    placeholder = np.zeros(len(data["sequence"]), dtype=int)
    loader = DataLoader(_dataset(data, placeholder), batch_size=batch_size, shuffle=False)
    logits_parts, similarity_parts, embedding_parts = [], [], []
    model.eval()
    with torch.no_grad():
        for sequence, mask, aggregate, signatures, _ in loader:
            logits, embeddings, similarities = model(sequence, mask, aggregate, signatures)
            logits_parts.append(logits.numpy())
            similarity_parts.append(similarities.numpy())
            embedding_parts.append(embeddings.numpy())
    return (
        np.concatenate(logits_parts), np.concatenate(similarity_parts),
        np.concatenate(embedding_parts),
    )
