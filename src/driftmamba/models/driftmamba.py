"""Causal selective-state flow encoder with path-signature fusion and prototypes."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class SelectiveStateBlock(nn.Module):
    """Portable Mamba-inspired selective state-space block.

    This pure-PyTorch scan is intentionally used on Windows where the fused `mamba-ssm` CUDA
    kernels are unavailable. It preserves causal, input-selective state updates and linear sequence
    complexity without claiming binary equivalence to the reference Mamba implementation.
    """

    def __init__(self, dimension: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dimension)
        self.input_projection = nn.Linear(dimension, dimension * 4)
        self.log_decay = nn.Parameter(torch.zeros(dimension))
        self.output_projection = nn.Linear(dimension, dimension)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        normalized = self.norm(inputs)
        candidate, write, read, step = self.input_projection(normalized).chunk(4, dim=-1)
        state = torch.zeros_like(inputs[:, 0])
        outputs = []
        decay_rate = F.softplus(self.log_decay).unsqueeze(0)
        for position in range(inputs.shape[1]):
            delta = F.softplus(step[:, position])
            decay = torch.exp(-decay_rate * delta)
            proposed = decay * state + (1.0 - decay) * torch.sigmoid(
                write[:, position]
            ) * torch.tanh(candidate[:, position])
            valid = mask[:, position].unsqueeze(-1)
            state = torch.where(valid, proposed, state)
            outputs.append(torch.sigmoid(read[:, position]) * state)
        scanned = torch.stack(outputs, dim=1)
        return inputs + self.dropout(self.output_projection(scanned))


class DriftMambaClassifier(nn.Module):
    def __init__(self, aggregate_dimension: int, number_classes: int, *,
                 model_dimension: int = 64, embedding_dimension: int = 64,
                 blocks: int = 3, prototypes_per_class: int = 2, signature_dimension: int = 6):
        super().__init__()
        self.number_classes = number_classes
        self.prototypes_per_class = prototypes_per_class
        self.packet_projection = nn.Linear(3, model_dimension)
        self.blocks = nn.ModuleList([SelectiveStateBlock(model_dimension) for _ in range(blocks)])
        self.aggregate_projection = nn.Sequential(
            nn.Linear(aggregate_dimension, model_dimension), nn.GELU(), nn.LayerNorm(model_dimension)
        )
        self.signature_projection = nn.Sequential(
            nn.Linear(signature_dimension, model_dimension // 2), nn.GELU()
        )
        fusion_dimension = model_dimension * 2 + model_dimension // 2
        self.embedding = nn.Sequential(
            nn.Linear(fusion_dimension, embedding_dimension), nn.GELU(),
            nn.LayerNorm(embedding_dimension), nn.Linear(embedding_dimension, embedding_dimension),
        )
        self.prototypes = nn.Parameter(
            torch.randn(number_classes, prototypes_per_class, embedding_dimension) * 0.02
        )
        self.log_temperature = nn.Parameter(torch.tensor(-2.3))

    def forward(self, sequence: torch.Tensor, mask: torch.Tensor, aggregate: torch.Tensor,
                signatures: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        encoded = self.packet_projection(sequence)
        for block in self.blocks:
            encoded = block(encoded, mask)
        weights = mask.unsqueeze(-1).to(encoded.dtype)
        mean_pool = (encoded * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)
        last_indices = mask.sum(dim=1).clamp_min(1) - 1
        last_pool = encoded[torch.arange(len(encoded), device=encoded.device), last_indices]
        fused = torch.cat([
            mean_pool + last_pool,
            self.aggregate_projection(aggregate),
            self.signature_projection(signatures),
        ], dim=-1)
        embedding = F.normalize(self.embedding(fused), dim=-1)
        prototypes = F.normalize(self.prototypes, dim=-1)
        similarities = torch.einsum("bd,ckd->bck", embedding, prototypes)
        class_similarity = similarities.max(dim=-1).values
        temperature = self.log_temperature.exp().clamp_min(0.01)
        return class_similarity / temperature, embedding, class_similarity


class FlowTransformerClassifier(DriftMambaClassifier):
    """Causal Transformer with DriftMamba's multi-view prototype classification head."""

    def __init__(self, aggregate_dimension: int, number_classes: int, *,
                 model_dimension: int = 64, embedding_dimension: int = 64,
                 blocks: int = 3, prototypes_per_class: int = 2, signature_dimension: int = 6):
        super().__init__(
            aggregate_dimension, number_classes, model_dimension=model_dimension,
            embedding_dimension=embedding_dimension, blocks=blocks,
            prototypes_per_class=prototypes_per_class, signature_dimension=signature_dimension,
        )
        layer = nn.TransformerEncoderLayer(
            model_dimension, nhead=4, dim_feedforward=model_dimension * 4, dropout=0.1,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=blocks, enable_nested_tensor=False
        )
        self.position = nn.Parameter(torch.randn(1, 64, model_dimension) * 0.02)

    def forward(self, sequence: torch.Tensor, mask: torch.Tensor, aggregate: torch.Tensor,
                signatures: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        encoded = self.packet_projection(sequence) + self.position[:, :sequence.shape[1]]
        causal = torch.triu(torch.ones(
            sequence.shape[1], sequence.shape[1], device=sequence.device, dtype=torch.bool
        ), diagonal=1)
        encoded = self.transformer(encoded, mask=causal, src_key_padding_mask=~mask)
        weights = mask.unsqueeze(-1).to(encoded.dtype)
        mean_pool = (encoded * weights).sum(1) / weights.sum(1).clamp_min(1)
        last_indices = mask.sum(1).clamp_min(1) - 1
        last_pool = encoded[torch.arange(len(encoded), device=encoded.device), last_indices]
        fused = torch.cat([
            mean_pool + last_pool, self.aggregate_projection(aggregate),
            self.signature_projection(signatures),
        ], dim=-1)
        embedding = F.normalize(self.embedding(fused), dim=-1)
        similarities = torch.einsum(
            "bd,ckd->bck", embedding, F.normalize(self.prototypes, dim=-1)
        )
        class_similarity = similarities.max(-1).values
        return class_similarity / self.log_temperature.exp().clamp_min(0.01), embedding, class_similarity


class HyenaFlowBlock(nn.Module):
    """Hyena-inspired causal gated long-convolution block for short packet sequences."""

    def __init__(self, dimension: int, dilation: int):
        super().__init__()
        self.norm = nn.LayerNorm(dimension)
        self.gates = nn.Linear(dimension, dimension * 2)
        self.filter = nn.Conv1d(
            dimension, dimension, kernel_size=5, dilation=dilation,
            padding=4 * dilation, groups=dimension,
        )
        self.mix = nn.Linear(dimension, dimension)
        self.dropout = nn.Dropout(0.1)

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        normalized = self.norm(inputs)
        value, gate = self.gates(normalized).chunk(2, dim=-1)
        filtered = self.filter(value.transpose(1, 2))[..., :inputs.shape[1]].transpose(1, 2)
        update = torch.sigmoid(gate) * filtered * mask.unsqueeze(-1)
        return inputs + self.dropout(self.mix(update))


class HyenaFlowClassifier(DriftMambaClassifier):
    """Prototype classifier using gated dilated long convolutions instead of attention/state scans."""

    def __init__(self, aggregate_dimension: int, number_classes: int, *,
                 model_dimension: int = 64, embedding_dimension: int = 64,
                 blocks: int = 3, prototypes_per_class: int = 2, signature_dimension: int = 6):
        super().__init__(
            aggregate_dimension, number_classes, model_dimension=model_dimension,
            embedding_dimension=embedding_dimension, blocks=blocks,
            prototypes_per_class=prototypes_per_class, signature_dimension=signature_dimension,
        )
        self.blocks = nn.ModuleList([
            HyenaFlowBlock(model_dimension, dilation=2**index) for index in range(blocks)
        ])


class StabilizedScalarLSTMBlock(nn.Module):
    """xLSTM-inspired scalar-memory block with stabilized exponential gates."""

    def __init__(self, dimension: int):
        super().__init__()
        self.norm = nn.LayerNorm(dimension)
        self.projection = nn.Linear(dimension * 2, dimension * 4)
        self.output = nn.Linear(dimension, dimension)

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        cell = torch.zeros_like(inputs[:, 0])
        normalizer = torch.zeros_like(cell)
        hidden = torch.zeros_like(cell)
        stabilizer = torch.zeros_like(cell)
        outputs = []
        for position in range(inputs.shape[1]):
            current = self.norm(inputs[:, position])
            candidate, input_log, forget_log, output_gate = self.projection(
                torch.cat([current, hidden], dim=-1)
            ).chunk(4, dim=-1)
            new_stabilizer = torch.maximum(forget_log + stabilizer, input_log)
            forget = torch.exp(forget_log + stabilizer - new_stabilizer)
            write = torch.exp(input_log - new_stabilizer)
            proposed_cell = forget * cell + write * torch.tanh(candidate)
            proposed_norm = forget * normalizer + write
            proposed_hidden = torch.sigmoid(output_gate) * proposed_cell / proposed_norm.clamp_min(1)
            valid = mask[:, position].unsqueeze(-1)
            cell = torch.where(valid, proposed_cell, cell)
            normalizer = torch.where(valid, proposed_norm, normalizer)
            hidden = torch.where(valid, proposed_hidden, hidden)
            stabilizer = torch.where(valid, new_stabilizer, stabilizer)
            outputs.append(hidden)
        return inputs + self.output(torch.stack(outputs, dim=1))


class XLSTMFlowClassifier(DriftMambaClassifier):
    """Flow classifier with portable xLSTM-inspired stabilized recurrent blocks."""

    def __init__(self, aggregate_dimension: int, number_classes: int, *,
                 model_dimension: int = 64, embedding_dimension: int = 64,
                 blocks: int = 3, prototypes_per_class: int = 2, signature_dimension: int = 6):
        super().__init__(
            aggregate_dimension, number_classes, model_dimension=model_dimension,
            embedding_dimension=embedding_dimension, blocks=blocks,
            prototypes_per_class=prototypes_per_class, signature_dimension=signature_dimension,
        )
        self.blocks = nn.ModuleList([
            StabilizedScalarLSTMBlock(model_dimension) for _ in range(blocks)
        ])


def prototype_loss(logits: torch.Tensor, class_similarity: torch.Tensor,
                   targets: torch.Tensor, margin: float = 0.15) -> torch.Tensor:
    cross_entropy = F.cross_entropy(logits, targets)
    correct = class_similarity.gather(1, targets.unsqueeze(1)).squeeze(1)
    wrong = class_similarity.masked_fill(
        F.one_hot(targets, class_similarity.shape[1]).bool(), -1.0
    ).max(dim=1).values
    hard_negative_margin = F.relu(margin + wrong - correct).mean()
    return cross_entropy + 0.25 * hard_negative_margin
