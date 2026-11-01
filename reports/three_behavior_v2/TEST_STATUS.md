# V2 Independent Test availability

**Status: `TEST_NOT_AVAILABLE`.** No independent, completeness-verified three-class Test set can be formed from the two source ZIPs without reusing V2 Train/Validation or treating missing annotations as negatives.

## Source audit

- Classroom Behavior ZIP: 64 images in its source `test` split. After the explicit mapping (`sleep` → sleeping; `hand` remains unmapped), those images contain **0 mapped target-positive images and 0 target instances**. The sole source scene proxy `CB_VIDEO:giua_mp4` is disjoint from the V2 Train/Validation scene proxies. Against all 1,300 V2 Train/Validation images, the 64 source-Test images have **0 exact SHA-256 matches** and **0 pHash matches at Hamming distance ≤4**. This does not make them a usable Test: target classes are absent, `hand` cannot be mapped, and source annotation completeness is unverified.
- Classroom Student v16 ZIP: source Train only; it has no Validation or Test split. Its unused groups remain source-Train data with annotation completeness unverified, so holding them out would not establish an independent, trusted Test.
- V2 Train/Validation uses 1,086/214 images across the accepted source manifest. Its existing audit found zero shared scene-group IDs, zero exact duplicates, and zero pHash≤4 pairs across those two splits. That does not authorize reusing either split as Test.

No image was moved, no Test labels were created, and no Validation image is designated as Test. Original ZIPs and the V2 Train/Validation dataset were not modified. Checkpoint integrity was preserved: official `best.pt` SHA-256 `6821a02bbe1dde7deea8a7936766d93a99bc07b6b77db804ade563d497492965`; V1 SHA-256 `b916dc1d075578626dbd6b94c777b6fc630eafea6d8d570417bb0e507c844a55`.

Use `HTTM_ThreeBehavior_V2_Kaggle_Test_Evaluation.ipynb` when a separately prepared Test package exists. Until then, do not report V2 Validation metrics as independent final evaluation.
