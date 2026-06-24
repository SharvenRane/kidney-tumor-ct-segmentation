# Predetermined Change Control Plan (Illustrative)

> **This is an illustrative, educational document for a portfolio project. It is
> not a regulatory submission and describes a hypothetical device.** It is
> structured to follow the FDA **final** guidance *"Marketing Submission
> Recommendations for a Predetermined Change Control Plan for Artificial
> Intelligence-Enabled Device Software Functions"* (issued December 4, 2024;
> docket FDA-2022-D-2628), whose statutory basis is section 515C of the FD&C Act
> (added by FDORA 2022).

A PCCP lets a manufacturer pre-specify, and have FDA authorize, a set of future
modifications to an AI-enabled device together with the protocol for making and
validating them, so that changes consistent with the authorized plan do not each
require a new marketing submission. Per the guidance, a PCCP has three
components: a Description of Modifications, a Modification Protocol, and an Impact
Assessment.

## Device context (hypothetical)

- **Device:** Kidney and renal-mass segmentation software for contrast-enhanced
  abdominal CT. Output is a voxel-level labelmap (kidney, mass) plus derived
  volumes, intended to assist a radiologist (not to autonomously diagnose).
- **Intended use / indications:** quantitative assistance for delineation of
  kidneys and renal masses on portal/nephrogenic-phase CT in adults; a
  concurrent-read aid, radiologist remains responsible for interpretation.
- **Plausible classification:** computer-assisted detection/segmentation, which
  would generally fall under 21 CFR 892.2090 (Class II), cleared via 510(k). The
  device described here is hypothetical; no clearance is claimed.

## Component 1 — Description of Modifications

Each modification is specific, verifiable, keeps the device within its intended
use, and states whether it is implemented manually or automatically and applied
globally or locally.

| # | Modification | Manual/Auto | Global/Local | Stays within intended use |
|---|---|---|---|---|
| M1 | Retrain on additional same-distribution CT to improve Dice/HD95 | Manual | Global | Yes |
| M2 | Add support for a new CT scanner manufacturer/model (same input type, contrast CT) | Manual | Global | Yes |
| M3 | Re-tune the operating point (probability threshold) to adjust the sensitivity/specificity trade-off within pre-set bounds | Manual | Global | Yes |

No modification expands the device to new anatomy, new modality, a pediatric
population, or autonomous diagnosis; those would require a separate submission.

## Component 2 — Modification Protocol

For every modification above, the protocol addresses the four areas the guidance
requires. The traceability table at the end links each modification to these
elements.

### 2.1 Data management practices
- Sources: de-identified contrast CT with clinician reference segmentations.
- Curation: inclusion/exclusion by phase, slice thickness, and reconstruction
  kernel; duplicate and corrupted-series screening.
- Sequestration: a permanently held-out test set, patient-level disjoint from
  all training and tuning data (GMLP Principle 4); no patient appears in more
  than one partition.
- Representativeness: monitor and report distribution of sex, age band, and
  scanner manufacturer/model (GMLP Principle 3; CLAIM items 33/34).

### 2.2 Re-training practices
- Fixed architecture (3D SegResNet), fixed preprocessing (1.5 mm isotropic,
  HU window [-200, 300]), fixed loss (Dice + cross-entropy), seeded runs.
- Version every training dataset and every resulting model; record the full
  environment (the MONAI Bundle `metadata.json` pins the version triple).

### 2.3 Performance evaluation (acceptance criteria)
Each candidate model is evaluated on the sequestered test set and must meet
**pre-specified acceptance criteria before release**:
- Kidney Dice ≥ the currently deployed model, within a non-inferiority margin.
- Tumor (mass) Dice ≥ the currently deployed model, within a non-inferiority
  margin, with the lower bound of the 95% CI not falling below the original
  cleared performance floor.
- No subgroup (sex, age band, scanner manufacturer) may regress by more than a
  pre-set margin versus the deployed model (fairness guardrail).
- HD95 not worse than the deployed model beyond a pre-set tolerance.
For M3 (operating-point change) the floor is expressed directly on sensitivity
and specificity, mirroring the imaging example in the guidance (retrain to
improve provided sensitivity and specificity do not drop below a set level).

### 2.4 Update procedures
- Staged rollout with versioned release notes; the cleared label is updated to
  reflect the active version (the guidance requires labeling to state that the
  device has an authorized PCCP and to be kept current).
- Rollback to the previous validated version if post-deployment monitoring
  (see `src/monitor.py`) trips a trigger.

## Component 3 — Impact Assessment

- **Benefit/risk per modification:** M1 and M2 expand robustness and reach with
  the main risk being silent regression on an unseen subgroup, mitigated by the
  sequestered test set and the subgroup non-regression guardrail. M3 trades
  sensitivity against specificity, mitigated by the pre-set bounds.
- **Collective impact:** modifications are evaluated individually against the
  unmodified device and the cumulative effect (e.g., repeated retraining drift)
  is bounded by always comparing to the original cleared performance floor, not
  only to the immediately prior version.
- **Effect on other functions:** segmentation is the only AI function; no drug
  or hardware constituent is affected.
- **Real-world performance link:** the Modification Protocol's triggers are the
  thresholds implemented in the monitoring module (PSI > 0.25, embedding-MMD
  shift p < 0.05, OOD fraction > 10%); crossing them initiates the retraining
  cycle described above. This is the operational realization of GMLP Principle 10.

## Traceability

| Modification | Data mgmt | Re-training | Performance eval | Update procedure |
|---|---|---|---|---|
| M1 | 2.1 | 2.2 | 2.3 (Dice/HD95 non-inferiority + subgroup guardrail) | 2.4 |
| M2 | 2.1 (new-scanner sampling) | 2.2 | 2.3 (per-manufacturer subgroup) | 2.4 |
| M3 | 2.1 | n/a (no retrain) | 2.3 (sens/spec bounds) | 2.4 |

---

*References: FDA PCCP final guidance (Dec 2024, FDA-2022-D-2628); GMLP Guiding
Principles (FDA/Health Canada/MHRA, Oct 2021). See `docs/GMLP_MAPPING.md`.*
