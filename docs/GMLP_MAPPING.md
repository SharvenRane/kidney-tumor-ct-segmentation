# GMLP Principles Mapping

How this project addresses each of the ten **Good Machine Learning Practice for
Medical Device Development: Guiding Principles** (FDA, Health Canada, MHRA,
October 2021). This is an educational mapping for a portfolio project, not a
claim of regulatory compliance.

| # | GMLP Principle | How this project addresses it |
|---|---|---|
| 1 | Multi-disciplinary expertise across the total product life cycle | Intended use, the human-in-the-loop concurrent-read framing, and clinical risk are stated up front in the model card and PCCP; the design treats segmentation as radiologist assistance, not autonomous diagnosis. |
| 2 | Good software engineering and security practices | Versioned, seeded, reproducible pipeline; packaged as a schema-validated MONAI Bundle with a pinned environment; deterministic data conversion; unit tests for the data, metrics, and monitoring code. |
| 3 | Clinical study participants and data sets are representative | Subgroup distribution (sex, age band, scanner manufacturer/model, site) is extracted from DICOM metadata and reported; performance is stratified across these groups in the validation report. |
| 4 | Training data sets are independent of test sets | Patient-level split: no patient appears in more than one of train/validation/test. The test set is held out and only touched for the final evaluation. |
| 5 | Selected reference datasets are based on best available methods | Reference standard is the clinician DICOM-SEG annotation distributed with C4KC-KiTS; its semiautomatic origin and inter-reader variability are stated as a limitation. |
| 6 | Model design is tailored to the data and intended use | 3D SegResNet at 1.5 mm isotropic with an abdominal HU window suited to contrast CT of kidney and renal mass; patch-based training and AMP fit the data and the available single-GPU hardware; dropout and augmentation mitigate overfitting. |
| 7 | Focus on the performance of the human-AI team | The device is framed as a concurrent-read aid; the model card states that the primary clinical evaluation would be a reader (MRMC) study of the radiologist-plus-AI team, which is described, not claimed. |
| 8 | Testing demonstrates device performance in clinically relevant conditions | Standalone performance is measured on the sequestered test set with Dice, HD95, sensitivity and specificity, each with 95% bootstrap confidence intervals, and broken out by subgroup. |
| 9 | Users are provided clear, essential information | The model card (CHAI Applied Model Card structure) and the one-page Model Facts label give intended use, inputs/outputs, performance with subgroup breakdown, known limitations and warnings; explainability overlays show what the model attends to. |
| 10 | Deployed models are monitored and re-training risks are managed | `src/monitor.py` implements embedding-drift (MMD, KS), output-score drift (PSI, JS) and Mahalanobis OOD detection with documented thresholds; those thresholds are the triggers in the PCCP Modification Protocol, closing the loop from monitoring to controlled re-training. |

---

*Reference: Good Machine Learning Practice for Medical Device Development:
Guiding Principles (FDA/Health Canada/MHRA, Oct 2021),
https://www.fda.gov/media/153486/download. See also `docs/PCCP.md`.*
