"""Paired-difference statistics: percentile bootstrap CI of the mean Δ_d."""

import numpy as np

RESAMPLES = 10_000
CHUNK = 500  # resamples drawn per batch, bounds memory at CHUNK × N indices


def bootstrap_ci(deltas, resamples=RESAMPLES, level=0.95, seed=0) -> tuple[float, float, float]:
    """(mean, lower, upper): resample the per-deal Δs with replacement, recompute the mean,
    take the (1-level)/2 and (1+level)/2 percentiles of those means."""
    x = np.asarray(deltas, dtype=float)
    rng = np.random.default_rng(seed)
    means = np.concatenate([x[rng.integers(0, len(x), size=(min(CHUNK, resamples - i), len(x)))].mean(axis=1)
                            for i in range(0, resamples, CHUNK)])
    tail = (1 - level) / 2
    lo, hi = np.quantile(means, [tail, 1 - tail])
    return float(x.mean()), float(lo), float(hi)
