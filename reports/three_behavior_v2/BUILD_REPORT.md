# HTTM ThreeBehavior V2 — Build Report

## Result

**BLOCKED_NO_CREDIBLE_TEST.** A group-held-out Train/Validation candidate package is ready for an experimental Kaggle run, but no Test split was created.

| Split | Images | sleeping boxes | using_phone boxes | raising_hand boxes |
|---|---:|---:|---:|---:|
| Train | 1,086 | 2,071 | 2,265 | 1,753 |
| Validation | 214 | 448 | 437 | 354 |
| Test | 0 | 0 | 0 | 0 |

Train is 83.5% and Validation is 16.5% of retained images. The requested 70/15/15 split could not be completed without inventing a Test set.

## Mapping and filtering

- `sleep` / `Sleeping` → `0 sleeping`.
- `Using Phone` → `1 using_phone`.
- `Hand Rising` → `2 raising_hand`.
- `hand`, reading, writing, and opaque classes were not mapped.
- Only images with at least one mapped positive were retained; empty labels were not created and no pretrained predictions were used as labels.
- 5,359 source images were quarantined or deduplicated: 2,584 had the ambiguous `hand` class; 2,324 had no mapped positive; 511 belonged to exact/near duplicate groups with conflicting target class profiles; 179 had empty/missing labels; 47 had invalid target bbox geometry; and 2 had opaque classes. Categories can overlap.
- 22 invalid non-target bbox rows were removed while otherwise usable images were retained.
- Exact/near duplicates with old Round3/Round4 pools were not used as an exclusion rule. The retained manifest includes 733 exact-overlap and 492 pHash-near-overlap source images identified by the prior audit.

## Limits

Annotation completeness is **UNVERIFIED**. Validation is a group-held-out diagnostic based on filename/date/camera-roll proxies joined with pHash distance ≤4; these are not verified capture-session IDs. Validation metrics must not be presented as final evaluation.

The Classroom Behavior ZIP's original Test split has no positives after the explicit mapping (`hand` remains unmapped). The Classroom Student v16 ZIP has only a Train split. A holdout fabricated from those training images would not establish an independent, completeness-verified Test set, so Test remains absent and the notebook skips Test evaluation.

## Artifacts

- Dataset: `ai/behavior_detection/datasets/HTTM_ThreeBehavior_V2/`
- Kaggle ZIP: `ai/behavior_detection/datasets/HTTM_ThreeBehavior_V2_Kaggle.zip`
- Notebook: `ai/behavior_detection/experiments/accuracy_first_development/HTTM_ThreeBehavior_V2_Kaggle.ipynb`
- Manifests: `source_manifest.csv`, `scene_groups.csv`, `quarantine_manifest.csv`, `class_mapping.csv` inside the dataset folder and ZIP.
- Training was not run locally.
