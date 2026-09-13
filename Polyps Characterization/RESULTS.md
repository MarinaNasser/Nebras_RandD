# ERCPMP JNET baseline — 2026-09-13

## Result figures

Patient-level metrics; confusion matrix rows are true labels and columns are
predicted labels. PNGs are saved at 300 DPI, with PDF copies in the same folder.
Regenerate with `python 'Polyps Characterization/plot_results.py'` from the repository root.

![Confusion matrix](outputs/ercpmp_jnet/figures/confusion_matrix.png)

![Training curves](outputs/ercpmp_jnet/figures/training_curves.png)

![Per-class performance](outputs/ercpmp_jnet/figures/per_class_metrics.png)

![Test summary](outputs/ercpmp_jnet/figures/test_summary.png)

## Evaluation

Training completed on the local RTX 5070 GPU. ResNet-18 was initialized from
ImageNet and trained on crops from the existing SegFormer-B3 checkpoint.

394 JPGs were audited; 302 images from 70 patients had valid JNET labels in the
spreadsheet. The other 92 images were excluded. Every included image produced a
segmentation component meeting the crop rule. Patient-disjoint splits contained
204/50/48 images from 44/13/13 patients (train/validation/test). There were no
byte-identical source images across splits.

Early stopping ended training after 26 epochs. Epoch 16 was selected using
validation patient macro F1 (0.4792). Test patients were evaluated only after
checkpoint selection; image probabilities were averaged per patient.

| Held-out patient metric | Result |
|---|---:|
| Accuracy | 30.77% (4/13) |
| Macro F1 | 0.2917 |
| Balanced accuracy (macro recall) | 36.90% |
| Majority-class accuracy baseline | 53.85% |

| JNET | Test patients | Recall | F1 |
|---|---:|---:|---:|
| 1 | 2 | 1.0000 | 0.6667 |
| 2A | 7 | 0.1429 | 0.2500 |
| 2B | 3 | 0.3333 | 0.2500 |
| 3 | 1 | 0.0000 | 0.0000 |

This baseline underperforms majority-class accuracy and is not ready for
deployment. Training loss improved substantially while validation fluctuated,
consistent with overfitting on this small patient cohort. Class 3 has only one
test patient. No claim of clinical diagnostic performance or external
generalization is supported. JNET predictions are optical labels, not pathology
diagnoses. Segmentation coverage does not establish localization accuracy.

Artifacts: `outputs/ercpmp_jnet/best.pt`, `history.json`, `test_results.json`,
`manifest.json`, `excluded.json`, `data_summary.json`, `environment.json`, and
`crop_review.jpg`. The checkpoint includes the source segmentation SHA-256.
The unit checks cover patient split isolation and crop/mask behavior.

Future improvement should prioritize auditing ambiguous/missing labels and
collecting more independently labeled patients. Compare alternatives using
patient-grouped validation or cross-validation within the development cohort;
do not tune against this reported test set.
