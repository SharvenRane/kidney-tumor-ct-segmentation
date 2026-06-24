# Model Card: Kidney + Tumor CT Segmentation

> Illustrative clinical model card for a portfolio project, structured on the CHAI Applied Model Card and FDA's example model card (Jan 2025 lifecycle **draft** guidance, not for implementation). Not a regulatory submission; describes a hypothetical device.


## Basic information

- **Model name:** Kidney + Renal-Mass CT Segmentation
- **Version:** see MONAI Bundle `configs/metadata.json`
- **Model type:** 3D convolutional neural network (SegResNet), semantic segmentation, 3 classes (background, kidney, mass)
- **Developer:** portfolio project (Sharven Rane)


## Release information

- **Regulatory status:** none; illustrative/educational only.
- **Inputs:** contrast-enhanced abdominal CT series (DICOM).
- **Outputs:** voxel labelmap (kidney, mass) + derived volumes.


## Uses and directions

- **Intended use:** quantitative assistance for delineation of kidneys and renal masses on contrast CT in adults, as a concurrent-read aid.
- **Intended users:** radiologists / imaging scientists.
- **Target population:** adults undergoing contrast-enhanced abdominal CT.
- **Out-of-scope:** non-contrast CT, other anatomy or modalities, pediatric patients, and any autonomous diagnostic use.


## Warnings

- Not a diagnostic device; a qualified clinician must review all output.
- Validated on a single public collection; performance on other scanners, sites, protocols and populations is unverified.
- Renal-mass (tumor) segmentation is the harder task and carries lower, more variable performance than kidney; see metrics below.
- Reference standard is semiautomatic clinician annotation with inherent inter-reader variability.


## Key metrics (standalone performance)

Held-out test cases: 32. Reference standard: clinician DICOM-SEG.


| Class | Dice (95% CI) | HD95 mm (95% CI) | Sensitivity (95% CI) | Specificity (95% CI) |
|---|---|---|---|---|
| kidney | 0.9201 (0.8892-0.9459) | 14.2059 (4.0-29.9191) | 0.9167 (0.8722-0.9511) | 0.9995 (0.9993-0.9997) |
| tumor | 0.6687 (0.5769-0.755) | 37.3941 (20.4134-57.9997) | 0.6582 (0.5697-0.742) | 0.9997 (0.9994-0.9999) |

Fairness / subgroup performance (Dice) is reported in `docs/VALIDATION_REPORT.md`, stratified by sex, age band and scanner manufacturer, with best- and worst-performing subgroups identified.


## Trust ingredients (AI system facts)

- **Architecture:** SegResNet, 3D, trained from scratch.
- **Training data:** C4KC-KiTS (TCIA), contrast abdominal CT with clinician kidney/mass DICOM-SEG; patient-level train/val/test split.
- **Preprocessing:** RAS orientation, 1.5 mm isotropic resampling, HU window [-200, 300], foreground crop.
- **Bias mitigation:** subgroup performance monitored; non-regression guardrails defined in the PCCP.
- **Ongoing maintenance:** drift monitoring (`src/monitor.py`) with thresholds wired to the PCCP Modification Protocol.


## Validation and evidence

- **Standalone (analytical):** performed; table above + validation report.
- **Clinical (human-AI team):** an MRMC reader study would be the primary clinical evaluation per FDA's draft guidance; described, not performed.


## Resources

- `docs/VALIDATION_REPORT.md`, `docs/PCCP.md`, `docs/GMLP_MAPPING.md`, `docs/MODEL_FACTS.md`, and the MONAI Bundle under `bundle/`.

