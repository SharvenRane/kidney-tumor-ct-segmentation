# Model Facts: Kidney + Renal-Mass CT Segmentation

> One-page clinician-facing label (Sendak et al., npj Digital Medicine 2020). Illustrative portfolio artifact.


**Summary:** Assists delineation of kidneys and renal masses on contrast-enhanced abdominal CT. Concurrent-read aid; not a diagnostic device.


**Mechanism:** A 3D SegResNet takes a CT series and outputs a per-voxel labelmap (kidney, mass) and derived volumes.


**Validation and performance** (n=32 held-out cases, vs clinician reference):

- Kidney Dice 0.9201 (0.8892-0.9459); mass Dice 0.6687 (0.5769-0.755).
- HD95: kidney 14.2059 (4.0-29.9191) mm, mass 37.3941 (20.4134-57.9997) mm.
- Subgroup breakdown by sex, age and scanner: see validation report.


**Uses and directions:** Adults, contrast abdominal CT. Review and edit all output before use. Not for non-contrast CT, other anatomy, or pediatric cases.


**Warnings:** Single-collection validation; performance unverified on other scanners/sites. Mass segmentation is less reliable than kidney. A clinician is responsible for all interpretation.


**Other information:** Monitored for drift in production; updates governed by a Predetermined Change Control Plan.

