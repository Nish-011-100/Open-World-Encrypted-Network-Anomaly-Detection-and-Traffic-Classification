# Experiment Results and Validation Status

These are genuine smoke-benchmark results from the official CESNET-QUIC22 XS edition. Synthetic
unit-test performance is never promoted as a research result.

## Verification completed

- Wireshark packet-to-flow conversion: 117,084 packets to 1,540 bidirectional flows.
- Automated tests: 11 passing.
- Static analysis: passing.
- Classical and deep training entry points: installed.
- Frozen-artifact inference: tested.
- VS Code inference, quick-training, report, test, and lint configurations: validated.
- Official DataZoo integration: installed and export command available.
- Official XS file verified at 2,714,067,131 bytes.
- Chronological smoke split: 5,000 train, 1,000 calibration, 1,000 known-test, and 1,000 unknown-test
  flows; 15 known and 58 held-out unknown applications.

## Genuine smoke-benchmark results

### Final four neural pipelines

| Pipeline | Accuracy | Balanced accuracy | Macro-F1 | Unknown AUROC | Primary strength |
|---|---:|---:|---:|---:|---|
| Heterogeneous deep ensemble | **0.823** | **0.7191** | **0.7264** | 0.6433 | Best overall classification |
| Causal Transformer | 0.806 | 0.7046 | 0.6981 | 0.6090 | Raw accuracy |
| xLSTM-inspired | 0.799 | 0.7087 | 0.6901 | 0.5643 | Early packet classification |
| DriftMamba (12 epochs) | 0.755 | 0.6253 | 0.6435 | **0.7011** | Unknown separation |

The ensemble uses majority application voting and averages calibrated known-traffic p-values across
DriftMamba-12, Transformer, and xLSTM-inspired. It is a heterogeneous neural ensemble, not three
random initializations of one architecture.

| Model | Known macro-F1 | Balanced accuracy | Unknown AUROC | AUPR-Unknown | FPR@95 | ECE |
|---|---:|---:|---:|---:|---:|---:|
| Random Forest | 0.6495 | 0.6614 | 0.7493 | 0.6810 | 0.6050 | 0.1101 |
| HistGradientBoosting | 0.6578 | 0.6730 | 0.7081 | 0.6576 | 0.6940 | 0.1759 |
| DriftMamba (12 epochs) | 0.6435 | 0.6253 | 0.7011 | 0.6671 | 0.8070 | 0.2671 |
| DriftMamba (20 epochs, accuracy operating point) | **0.7006** | **0.6972** | 0.6357 | 0.6114 | 0.8560 | 0.2925 |
| Causal Flow Transformer (12 epochs) | 0.6981 | **0.7046** | 0.6090 | 0.6119 | 0.8930 | 0.3015 |
| xLSTM-inspired Flow Encoder (12 epochs) | 0.6901 | **0.7087** | 0.5643 | 0.5714 | 0.9200 | — |
| HyenaFlow (12 epochs) | 0.5605 | 0.5513 | 0.6794 | 0.6523 | 0.8310 | — |
| LOF-gated Random Forest | 0.6495 | — | 0.6938 | — | — | — |
| One-Class-SVM-gated Random Forest | 0.6495 | — | 0.7246 | — | — | — |
| RBF SVM | 0.1111 | — | 0.5403 | — | — | — |

DriftMamba reached 0.903 empirical known coverage at a 0.90 conformal target. Its 8/16/32-packet
macro-F1 values were 0.3893/0.5005/0.6435. Random Forest remains the strongest model on unknown
AUROC in this bounded run; no superiority claim is made.

The 20-epoch model raises accuracy to 0.777 and macro-F1 to 0.7006, but sacrifices unknown AUROC.
It is therefore an accuracy-oriented operating point, while the 12-epoch checkpoint is the more
balanced open-world operating point. Class-balanced sampling was also tested (macro-F1 0.6332,
unknown AUROC 0.6518) and was not retained as the default.

The causal Transformer reached the highest raw accuracy (0.806) and nearly matched DriftMamba's
macro-F1. It was also stronger at 8 packets (0.4626 macro-F1 versus DriftMamba's 0.3683), but its
unknown AUROC was lower. It is retained as the strongest early/closed-set neural baseline.

xLSTM-inspired achieved 0.799 accuracy and the best balanced accuracy, plus the strongest 8-packet
macro-F1 (0.5182), but weak unknown separation. HyenaFlow is weaker for known classification but
provides a useful gated-long-convolution contrast and 0.6794 unknown AUROC. Neither implementation
claims exact equivalence to the authors' optimized research packages.

## Legacy reference ablations (not headline models)

Isolation Forest and the dense autoencoder were evaluated only to reproduce and test ideas from the
reference GitHub repository. They achieved unknown AUROC 0.5748 and 0.5833 respectively and are no
longer part of the default benchmark or final model comparison. Their saved reports remain for
auditability; they are not proposed as final components.

Following the leakage-free design in khush1811/Network-Traffic-Anomaly-Detection, a dense numeric
autoencoder with Huber reconstruction loss, early stopping, robust calibration scaling, and
classifier-score fusion was evaluated. It did not improve open-world detection on QUIC application
holdouts, so it is retained as a reproducible ablation rather than promoted as the final detector.
Tesseract/EasyOCR are OCR tools and are intentionally excluded from packet/flow modeling; they are
only appropriate if a separate scanned-report or screenshot ingestion feature is required.

Additional model-family ablations cover local-density novelty detection (LOF), nonlinear support
estimation (One-Class SVM), and supervised RBF classification. One-Class SVM was the strongest
novelty gate but did not surpass the plain Random Forest confidence score. The RBF SVM performed
poorly on the imbalanced multimodal application classes and is not a final-model candidate.

## Acceptance criteria

- Conformal known coverage should be close to the configured 90% target.
- DriftMamba should be retained only if it improves a meaningful open-world metric or the early-flow
  accuracy/latency tradeoff over classical baselines.
- All final numbers must be generated from saved reports, not copied manually from transient output.
