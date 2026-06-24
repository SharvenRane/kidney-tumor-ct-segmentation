"""Production drift monitoring for the deployed segmentation model.

This is the concrete engineering realization of GMLP Principle 10 (monitor
deployed performance, manage retraining risk) and it supplies the trigger logic
for the PCCP Modification Protocol. Each detector below has a documented
threshold; crossing it is what would initiate a retraining cycle.

Detectors:
  * Embedding drift  - MMD (kernel two-sample test) + per-dimension KS on the
    model's pooled encoder features, train (reference) vs production. Imaging drift
    is best seen in feature space, not raw pixels (Rabanser 2019; Glocker 2023).
  * Output-score drift - Population Stability Index (PSI) and Jensen-Shannon
    divergence on the per-case predicted tumor-volume fraction.
  * Out-of-distribution - Mahalanobis distance in encoder-feature space with an
    SPC (3-sigma) control limit set on the reference distribution.

The pure functions are framework-light (numpy) so they are unit-testable without
a GPU or a trained model. extract_embeddings() and the CLI wire them to the model.
"""
import argparse
import json

import numpy as np

# Documented thresholds (the PCCP trigger table references these).
PSI_MODERATE, PSI_SIGNIFICANT = 0.10, 0.25  # >0.1 investigate, >0.25 retrain
KS_ALPHA = 0.05
MMD_PERMUTATIONS = 200


def population_stability_index(reference, current, bins=10):
    """PSI between two 1-D distributions. <0.1 stable, 0.1-0.25 moderate, >0.25 large."""
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(reference, quantiles))
    if len(edges) < 2:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    ref_pct = np.histogram(reference, bins=edges)[0] / len(reference)
    cur_pct = np.histogram(current, bins=edges)[0] / len(current)
    eps = 1e-6
    ref_pct = np.clip(ref_pct, eps, None)
    cur_pct = np.clip(cur_pct, eps, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def js_divergence(reference, current, bins=20):
    """Jensen-Shannon divergence in bits (0 identical, 1 maximally different)."""
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    lo = min(reference.min(), current.min())
    hi = max(reference.max(), current.max())
    edges = np.linspace(lo, hi, bins + 1)
    p = np.histogram(reference, bins=edges)[0].astype(float)
    q = np.histogram(current, bins=edges)[0].astype(float)
    p /= p.sum() + 1e-12
    q /= q.sum() + 1e-12
    m = 0.5 * (p + q)

    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def ks_statistic(reference, current):
    """Two-sample Kolmogorov-Smirnov D statistic and approximate p-value."""
    reference = np.sort(np.asarray(reference, dtype=float))
    current = np.sort(np.asarray(current, dtype=float))
    allv = np.concatenate([reference, current])
    cdf_r = np.searchsorted(reference, allv, side="right") / len(reference)
    cdf_c = np.searchsorted(current, allv, side="right") / len(current)
    d = float(np.max(np.abs(cdf_r - cdf_c)))
    n, m = len(reference), len(current)
    en = np.sqrt(n * m / (n + m))
    p = float(np.exp(-2 * (d * en) ** 2))
    return d, p


def _rbf_kernel(x, y, gamma):
    xx = np.sum(x * x, axis=1)[:, None]
    yy = np.sum(y * y, axis=1)[None, :]
    sq = xx + yy - 2 * x @ y.T
    return np.exp(-gamma * np.clip(sq, 0, None))


def mmd2(x, y, gamma=None):
    """Squared maximum mean discrepancy with an RBF kernel (median heuristic)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if gamma is None:
        pooled = np.vstack([x, y])
        d2 = np.sum((pooled[:, None, :] - pooled[None, :, :]) ** 2, axis=2)
        med = np.median(d2[d2 > 0]) if np.any(d2 > 0) else 1.0
        gamma = 1.0 / (med + 1e-12)
    kxx = _rbf_kernel(x, x, gamma)
    kyy = _rbf_kernel(y, y, gamma)
    kxy = _rbf_kernel(x, y, gamma)
    return float(kxx.mean() + kyy.mean() - 2 * kxy.mean()), gamma


def mmd_permutation_test(x, y, n_perm=MMD_PERMUTATIONS, seed=0):
    """Permutation p-value for the MMD two-sample test."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    observed, gamma = mmd2(x, y)
    pooled = np.vstack([x, y])
    n = len(x)
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_perm):
        rng.shuffle(pooled)
        stat, _ = mmd2(pooled[:n], pooled[n:], gamma=gamma)
        if stat >= observed:
            count += 1
    return observed, (count + 1) / (n_perm + 1)


def mahalanobis_ood(reference, current):
    """Per-sample Mahalanobis distance of `current` against the reference Gaussian,
    with a 3-sigma SPC control limit fit on the reference distances."""
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    mu = reference.mean(axis=0)
    cov = np.cov(reference, rowvar=False) + 1e-6 * np.eye(reference.shape[1])
    inv = np.linalg.inv(cov)

    def dist(m):
        diff = m - mu
        return np.sqrt(np.einsum("ij,jk,ik->i", diff, inv, diff))

    ref_d = dist(reference)
    cur_d = dist(current)
    control_limit = float(ref_d.mean() + 3 * ref_d.std())
    return cur_d, control_limit


def drift_report(ref_embeddings, cur_embeddings, ref_scores, cur_scores):
    """Run all detectors and apply documented thresholds to produce triggers."""
    mmd_val, mmd_p = mmd_permutation_test(ref_embeddings, cur_embeddings)
    ks_per_dim = [ks_statistic(ref_embeddings[:, j], cur_embeddings[:, j])
                  for j in range(ref_embeddings.shape[1])]
    n_ks_sig = sum(1 for _, p in ks_per_dim if p < KS_ALPHA / ref_embeddings.shape[1])
    psi = population_stability_index(ref_scores, cur_scores)
    js = js_divergence(ref_scores, cur_scores)
    cur_d, limit = mahalanobis_ood(ref_embeddings, cur_embeddings)
    frac_ood = float(np.mean(cur_d > limit))

    triggers = []
    if mmd_p < 0.05:
        triggers.append("embedding MMD shift (p<0.05): covariate/acquisition shift")
    if psi > PSI_SIGNIFICANT:
        triggers.append(f"output PSI {psi:.3f} > {PSI_SIGNIFICANT}: retrain/recalibrate")
    elif psi > PSI_MODERATE:
        triggers.append(f"output PSI {psi:.3f} > {PSI_MODERATE}: investigate")
    if frac_ood > 0.10:
        triggers.append(f"{frac_ood:.0%} of cases OOD (>3-sigma): review acquisition")
    return {
        "embedding_mmd": round(mmd_val, 6), "embedding_mmd_pvalue": round(mmd_p, 4),
        "embedding_ks_dims_significant": n_ks_sig,
        "embedding_ks_dims_total": ref_embeddings.shape[1],
        "output_psi": round(psi, 4), "output_js_divergence": round(js, 4),
        "ood_fraction": round(frac_ood, 4), "ood_control_limit": round(limit, 3),
        "triggers": triggers,
        "thresholds": {"psi_moderate": PSI_MODERATE, "psi_significant": PSI_SIGNIFICANT,
                       "ks_alpha": KS_ALPHA, "ood_sigma": 3},
    }


def extract_embeddings(model, loader, device):
    """Globally-pooled encoder features + predicted tumor-volume fraction per case."""
    import torch

    model.eval()
    embeddings, scores = [], []
    with torch.no_grad():
        for batch in loader:
            x = batch["image"].to(device)
            with torch.amp.autocast("cuda"):
                feats = model.encoder(x) if hasattr(model, "encoder") else None
                logits = model(x)
            if feats is None:
                feats = logits
            feat = feats[-1] if isinstance(feats, (list, tuple)) else feats
            pooled = feat.float().mean(dim=tuple(range(2, feat.ndim)))[0].cpu().numpy()
            embeddings.append(pooled)
            pred = torch.argmax(logits, dim=1)
            tumor_frac = float((pred == 2).sum()) / float(pred.numel())
            scores.append(tumor_frac)
    return np.array(embeddings), np.array(scores)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="C:/Users/sharv/data/c4kc-kits-nifti/dataset.json")
    ap.add_argument("--checkpoint", default="outputs/best.pt")
    ap.add_argument("--out", default="outputs/drift_report.json")
    args = ap.parse_args()
    import torch
    from monai.data import CacheDataset, DataLoader
    from monai.networks.nets import SegResNet
    from data import build_datalist, val_transforms

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split = build_datalist(args.dataset)
    model = SegResNet(spatial_dims=3, in_channels=1, out_channels=3, init_filters=16,
                      blocks_down=(1, 2, 2, 4), blocks_up=(1, 1, 1)).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    def loader_for(items):
        return DataLoader(CacheDataset(items, val_transforms(), cache_rate=0.0, num_workers=0),
                          batch_size=1, num_workers=0)

    ref_emb, ref_sc = extract_embeddings(model, loader_for(split["train"]), device)
    cur_emb, cur_sc = extract_embeddings(model, loader_for(split["test"]), device)
    report = drift_report(ref_emb, cur_emb, ref_sc, cur_sc)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
