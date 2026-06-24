"""Tests for the drift-monitoring detectors. Pure numpy, no model or GPU."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import monitor as M  # noqa: E402


def _rng():
    return np.random.default_rng(0)


def test_psi_low_when_same_distribution():
    r = _rng()
    a, b = r.normal(0, 1, 500), r.normal(0, 1, 500)
    assert M.population_stability_index(a, b) < 0.1


def test_psi_high_when_shifted():
    r = _rng()
    a, c = r.normal(0, 1, 500), r.normal(1.5, 1, 500)
    assert M.population_stability_index(a, c) > M.PSI_SIGNIFICANT


def test_js_divergence_bounds_and_ordering():
    r = _rng()
    a, b, c = r.normal(0, 1, 500), r.normal(0, 1, 500), r.normal(2, 1, 500)
    same, shift = M.js_divergence(a, b), M.js_divergence(a, c)
    assert 0.0 <= same <= 1.0 and 0.0 <= shift <= 1.0
    assert shift > same


def test_ks_detects_shift():
    r = _rng()
    a, b, c = r.normal(0, 1, 400), r.normal(0, 1, 400), r.normal(1.5, 1, 400)
    _, p_same = M.ks_statistic(a, b)
    d_shift, p_shift = M.ks_statistic(a, c)
    assert p_same > 0.05 and p_shift < 0.01 and d_shift > 0.2


def test_mmd_permutation_pvalue():
    r = _rng()
    x, y, z = r.normal(0, 1, (60, 8)), r.normal(0, 1, (60, 8)), r.normal(0.9, 1, (60, 8))
    _, p_same = M.mmd_permutation_test(x, y, n_perm=100)
    _, p_shift = M.mmd_permutation_test(x, z, n_perm=100)
    assert p_same > 0.1 and p_shift < 0.05


def test_mahalanobis_flags_out_of_distribution():
    r = _rng()
    x, z = r.normal(0, 1, (80, 6)), r.normal(3, 1, (80, 6))
    dist, limit = M.mahalanobis_ood(x, z)
    assert float((dist > limit).mean()) > 0.5


def test_drift_report_triggers_on_shift_only():
    r = _rng()
    x, z = r.normal(0, 1, (60, 8)), r.normal(1.0, 1, (60, 8))
    a, c = r.normal(0, 1, 400), r.normal(1.5, 1, 400)
    clean = M.drift_report(x, r.normal(0, 1, (60, 8)), a, r.normal(0, 1, 400))
    shifted = M.drift_report(x, z, a, c)
    assert clean["triggers"] == []
    assert len(shifted["triggers"]) >= 1
