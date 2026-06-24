# Validation Report: Kidney + Tumor CT Segmentation

> Illustrative analytical (standalone) performance evaluation for a portfolio project. Not a regulatory submission. Reference standard: clinician DICOM-SEG annotations from the C4KC-KiTS collection.


**Held-out test cases:** 32 (patient-level split, no patient in more than one partition).


## Standalone performance (algorithm vs reference standard)

| Class | Dice (95% CI) | HD95 mm (95% CI) | Sensitivity (95% CI) | Specificity (95% CI) |
|---|---|---|---|---|
| kidney | 0.9201 (0.8892-0.9459) | 14.2059 (4.0-29.9191) | 0.9167 (0.8722-0.9511) | 0.9995 (0.9993-0.9997) |
| tumor | 0.6687 (0.5769-0.755) | 37.3941 (20.4134-57.9997) | 0.6582 (0.5697-0.742) | 0.9997 (0.9994-0.9999) |

## Subgroup performance (Dice)

Per CLAIM items 33/34/36 and FUTURE-AI fairness: performance stratified by sex, age band and scanner manufacturer to surface best- and worst-performing subpopulations.


**By sex**

| sex | kidney | tumor |
|---|---|---|
| F | 0.9345 (n=14) | 0.6621 (n=14) |
| M | 0.909 (n=18) | 0.6738 (n=18) |

**By age_band**

| age_band | kidney | tumor |
|---|---|---|
| <50 | 0.9435 (n=7) | 0.7605 (n=7) |
| 50-64 | 0.9198 (n=17) | 0.6974 (n=17) |
| >=65 | 0.9004 (n=8) | 0.5275 (n=8) |

**By manufacturer**

| manufacturer | kidney | tumor |
|---|---|---|
| GE MEDICAL SYSTEMS | 0.9541 (n=5) | 0.7078 (n=5) |
| Philips | 0.9679 (n=1) | 0.8978 (n=1) |
| SIEMENS | 0.9075 (n=22) | 0.6589 (n=22) |
| TOSHIBA | 0.9353 (n=4) | 0.6162 (n=4) |

## Notes and limitations

- This is **analytical/standalone** performance only. Clinical validation of the human-AI team (an MRMC reader study, the primary evaluation for imaging aids per FDA's Jan 2025 draft guidance) is described in the model card, not performed here.

- The reference standard is semiautomatic clinician segmentation; inter-reader variability is a known source of label noise.

- The test set is single-collection; multi-site, multi-scanner external validation would be required before any clinical claim.

