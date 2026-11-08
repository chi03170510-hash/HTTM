# HTTM ThreeBehavior V2 — Split and Leakage Report

- Train: 1,086 images across scene groups; Validation: 214 images across held-out scene groups; Test: intentionally absent.
- Shared scene-group IDs across Train and Validation: **0**.
- Exact duplicates across Train and Validation: **0**.
- pHash distance ≤4 pairs across Train and Validation: **0** (checked exhaustively over retained images).
- Near-duplicate pHash links and shared metadata proxies were unioned before assigning groups; group assignment did not depend on the original source split.
- 937 exact/very-near duplicate representatives (pHash distance ≤2 or byte-identical) were removed. 511 further duplicate-group members were quarantined where their target-class profiles conflicted.
- No Round3/Round4 overlap filter was applied; 733 retained images are exact overlaps and 492 retained images are pHash-near overlaps with the prior local pool. Those older datasets are outside this V2 split and do not contribute images to it.
- Filename/video, date, camera-roll, and pHash grouping are proxies. Capture-session independence is not verified; pHash grouping prevents near-identical frames crossing Train/Validation but cannot prove scene identity.

## Test status

No credible independent Test split is available. The Classroom Behavior source's Test split has no mapped target positives, and the v16 source provides no Validation/Test split. Both sources lack approved full-image annotation-completeness review. V2 therefore has `test: null`; the Kaggle notebook trains/selects with the diagnostic Validation split and skips Test metrics. Status: **BLOCKED_NO_CREDIBLE_TEST**.
