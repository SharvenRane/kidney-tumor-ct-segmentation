"""Generate the clinical model card and a one-page Model Facts label from real
metrics. Run after evaluate.py so every number traces to outputs/metrics.json.

The model card follows the CHAI Applied Model Card structure (the de-facto US
clinical model card, aligned to ONC HTI-1 transparency and to FDA's example
model card in the Jan 2025 lifecycle draft guidance). The Model Facts label
follows Sendak et al. (npj Digital Medicine 2020): a one-page, clinician-facing
summary that explicitly includes Uses/Directions and Warnings.
"""
import argparse
import json
import os


def fmt(x):
    if not x:
        return "n/a"
    return f"{x['mean']} ({x['ci95'][0]}-{x['ci95'][1]})"


def model_card(m):
    L = []
    L.append("# Model Card: Kidney + Tumor CT Segmentation\n")
    L.append("> Illustrative clinical model card for a portfolio project, structured on the "
             "CHAI Applied Model Card and FDA's example model card (Jan 2025 lifecycle "
             "**draft** guidance, not for implementation). Not a regulatory submission; "
             "describes a hypothetical device.\n")

    L.append("\n## Basic information\n")
    L.append("- **Model name:** Kidney + Renal-Mass CT Segmentation\n"
             "- **Version:** see MONAI Bundle `configs/metadata.json`\n"
             "- **Model type:** 3D convolutional neural network (SegResNet), semantic "
             "segmentation, 3 classes (background, kidney, mass)\n"
             "- **Developer:** portfolio project (Sharven Rane)\n")

    L.append("\n## Release information\n")
    L.append("- **Regulatory status:** none; illustrative/educational only.\n"
             "- **Inputs:** contrast-enhanced abdominal CT series (DICOM).\n"
             "- **Outputs:** voxel labelmap (kidney, mass) + derived volumes.\n")

    L.append("\n## Uses and directions\n")
    L.append("- **Intended use:** quantitative assistance for delineation of kidneys and "
             "renal masses on contrast CT in adults, as a concurrent-read aid.\n"
             "- **Intended users:** radiologists / imaging scientists.\n"
             "- **Target population:** adults undergoing contrast-enhanced abdominal CT.\n"
             "- **Out-of-scope:** non-contrast CT, other anatomy or modalities, pediatric "
             "patients, and any autonomous diagnostic use.\n")

    L.append("\n## Warnings\n")
    L.append("- Not a diagnostic device; a qualified clinician must review all output.\n"
             "- Validated on a single public collection; performance on other scanners, "
             "sites, protocols and populations is unverified.\n"
             "- Renal-mass (tumor) segmentation is the harder task and carries lower, more "
             "variable performance than kidney; see metrics below.\n"
             "- Reference standard is semiautomatic clinician annotation with inherent "
             "inter-reader variability.\n")

    L.append("\n## Key metrics (standalone performance)\n")
    L.append(f"Held-out test cases: {m['n_test']}. Reference standard: clinician DICOM-SEG.\n")
    L.append("\n| Class | Dice (95% CI) | HD95 mm (95% CI) | Sensitivity (95% CI) | "
             "Specificity (95% CI) |")
    L.append("|---|---|---|---|---|")
    for c in m["classes"]:
        o = m["overall"][c]
        L.append(f"| {c} | {fmt(o['dice'])} | {fmt(o['hd95'])} | {fmt(o['sensitivity'])} "
                 f"| {fmt(o['specificity'])} |")
    L.append("\nFairness / subgroup performance (Dice) is reported in "
             "`docs/VALIDATION_REPORT.md`, stratified by sex, age band and scanner "
             "manufacturer, with best- and worst-performing subgroups identified.\n")

    L.append("\n## Trust ingredients (AI system facts)\n")
    L.append("- **Architecture:** SegResNet, 3D, trained from scratch.\n"
             "- **Training data:** C4KC-KiTS (TCIA), contrast abdominal CT with clinician "
             "kidney/mass DICOM-SEG; patient-level train/val/test split.\n"
             "- **Preprocessing:** RAS orientation, 1.5 mm isotropic resampling, HU window "
             "[-200, 300], foreground crop.\n"
             "- **Bias mitigation:** subgroup performance monitored; non-regression "
             "guardrails defined in the PCCP.\n"
             "- **Ongoing maintenance:** drift monitoring (`src/monitor.py`) with thresholds "
             "wired to the PCCP Modification Protocol.\n")

    L.append("\n## Validation and evidence\n")
    L.append("- **Standalone (analytical):** performed; table above + validation report.\n"
             "- **Clinical (human-AI team):** an MRMC reader study would be the primary "
             "clinical evaluation per FDA's draft guidance; described, not performed.\n")
    L.append("\n## Resources\n")
    L.append("- `docs/VALIDATION_REPORT.md`, `docs/PCCP.md`, `docs/GMLP_MAPPING.md`, "
             "`docs/MODEL_FACTS.md`, and the MONAI Bundle under `bundle/`.\n")
    return "\n".join(L) + "\n"


def model_facts(m):
    kid = m["overall"]["kidney"]
    tum = m["overall"]["tumor"]
    L = []
    L.append("# Model Facts: Kidney + Renal-Mass CT Segmentation\n")
    L.append("> One-page clinician-facing label (Sendak et al., npj Digital Medicine 2020). "
             "Illustrative portfolio artifact.\n")
    L.append("\n**Summary:** Assists delineation of kidneys and renal masses on "
             "contrast-enhanced abdominal CT. Concurrent-read aid; not a diagnostic device.\n")
    L.append("\n**Mechanism:** A 3D SegResNet takes a CT series and outputs a per-voxel "
             "labelmap (kidney, mass) and derived volumes.\n")
    L.append(f"\n**Validation and performance** (n={m['n_test']} held-out cases, vs clinician "
             "reference):\n")
    L.append(f"- Kidney Dice {fmt(kid['dice'])}; mass Dice {fmt(tum['dice'])}.\n"
             f"- HD95: kidney {fmt(kid['hd95'])} mm, mass {fmt(tum['hd95'])} mm.\n"
             "- Subgroup breakdown by sex, age and scanner: see validation report.\n")
    L.append("\n**Uses and directions:** Adults, contrast abdominal CT. Review and edit all "
             "output before use. Not for non-contrast CT, other anatomy, or pediatric cases.\n")
    L.append("\n**Warnings:** Single-collection validation; performance unverified on other "
             "scanners/sites. Mass segmentation is less reliable than kidney. A clinician is "
             "responsible for all interpretation.\n")
    L.append("\n**Other information:** Monitored for drift in production; updates governed by "
             "a Predetermined Change Control Plan.\n")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", default="outputs/metrics.json")
    ap.add_argument("--outdir", default="docs")
    args = ap.parse_args()
    m = json.load(open(args.metrics))
    os.makedirs(args.outdir, exist_ok=True)
    with open(os.path.join(args.outdir, "MODEL_CARD.md"), "w") as f:
        f.write(model_card(m))
    with open(os.path.join(args.outdir, "MODEL_FACTS.md"), "w") as f:
        f.write(model_facts(m))
    print(f"Wrote {args.outdir}/MODEL_CARD.md and {args.outdir}/MODEL_FACTS.md")


if __name__ == "__main__":
    main()
