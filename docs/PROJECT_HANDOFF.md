# Open-World Encrypted Network Anomaly Detection and Traffic Classification

**Project identifier:** NetAnomaly-OW

## One-line pitch

NetAnomaly-OW classifies encrypted QUIC applications without payload inspection, rejects unseen
applications with calibrated uncertainty, and detects when live traffic drifts away from training.

## Problem

Encrypted traffic classifiers are often evaluated as closed-set models: every test application is
assumed to have appeared during training. Production networks violate that assumption through new
applications, protocol evolution, changing user behavior, and new QUIC versions. Ordinary Softmax
confidence can remain high even when the input is unfamiliar.

## Proposed solution

The model combines three privacy-preserving views:

1. A causal selective state-space encoder learns from packet size, direction, and timing sequences.
2. A second-order path signature explicitly represents signed-volume/time trajectory geometry.
3. Robust-scaled aggregate flow statistics summarize the complete bidirectional flow.

The fused representation is mapped to a normalized embedding and compared with multiple learned
prototypes per application. Split-conformal calibration converts prototype nonconformity into a
known-traffic p-value and prediction set. Feature PSI and embedding-centroid movement monitor drift.

## Connection to the original ideation

The project retains the original proposal's context-aware flow embeddings, packet timing and size
paths, path signatures, contrastive separation, hard negatives, open-world recognition, real-time
prefix evaluation, and interpretable confidence. It advances the encoder from LSTM/Transformer-lite
to a portable selective state-space model and replaces synthetic-only unknown generation with real
complete-class holdouts and conformal rejection.

## Technical architecture

```text
CESNET QUIC flow
  |-- PPI sequence ------------> causal selective-state blocks --|
  |-- signed-size/time path ----> order-2 path signature --------|--> fused embedding
  |-- aggregate statistics -----> robust feature projection -----|
                                                                  |
                                       hyperspherical prototypes <-|
                                              |
                           application logits + prototype distance
                                              |
                              split-conformal calibration
                                              |
                  known class / prediction set / UNKNOWN / drift warning
```

## Dataset protocol

- Dataset: official CESNET-QUIC22 DataZoo XS edition.
- Train period: W-2022-44.
- Test period: W-2022-45.
- Known classes: 15 most frequent applications, grouped consistently by provider.
- Unknown classes: complete application classes excluded from training.
- Bounded first experiment: 100k train, 20k calibration, 20k known test, 20k unknown test.
- IP addresses are removed from canonical artifacts and are never model features.

## Model comparison

- Random Forest: strong nonlinear bagging baseline.
- Histogram Gradient Boosting: strong additive tree baseline.
- DriftMamba without signatures: sequence ablation.
- Path-signature projection without sequence encoder: mathematical-view ablation.
- Full DriftMamba: multi-view model with multiple class prototypes.

## Primary metrics

- Known traffic: macro-F1, balanced accuracy, per-class recall.
- Unknown traffic: AUROC, AUPR-Unknown, FPR at 95% unknown recall, unknown recall.
- Trust: conformal coverage, prediction-set size, Expected Calibration Error.
- Deployment: performance at 8/16/32/64 packets and inference latency.
- Drift: PSI and representation-centroid change by chronological window.

## Claims that may be made now

- The complete code path is implemented and tested.
- The project uses leakage-safe training/calibration/test artifacts.
- Unknown traffic consists of held-out application classes, not arbitrary anomalous packets.
- Inference produces calibrated p-values and prediction sets.
- Eleven automated tests and static checks pass.

## Claims that remain out of scope

- Superiority over ET-BERT, YaTC, TrafficGPT, or the classical baselines.
- Real-time suitability on target hardware.
- Robustness across networks beyond the evaluated CESNET periods.

## Reproduction

```powershell
.\scripts\run_cesnet_xs_experiment.ps1 -Epochs 15
jupyter lab notebooks
```

The DataZoo download is resumable. Generated data, models, and reports are ignored by Git.
