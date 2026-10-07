"""Eval harness: mirror test, rotations, bootstrap coverage, sealed-deal guard."""

import numpy as np
import pytest

from bigtwo.agents.greedy import GreedyAgent
from bigtwo.agents.heuristic import HeuristicAgent
from bigtwo.engine.deal import deal_stream
from bigtwo.engine.variants import HomeVariant, LiteratureVariant
from bigtwo.eval import gate
from bigtwo.eval.duplicate import FORMATS, deal_deltas, rotations
from bigtwo.eval.stats import bootstrap_ci


@pytest.mark.parametrize("variant", [LiteratureVariant(), HomeVariant(1)], ids=["literature", "home"])
@pytest.mark.parametrize("agent", [GreedyAgent, HeuristicAgent])
def test_mirror_gives_exactly_zero_on_every_deal(variant, agent):
    for deal in deal_stream(4, 1, 21, 50):
        assert deal_deltas(variant, deal, agent(), agent(), FORMATS) == dict.fromkeys(FORMATS, 0.0)


def test_rotations_cover_every_seat_equally():
    for fmt in FORMATS:
        seats = [s for r in rotations(fmt, 4) for s in r]
        assert {seats.count(s) for s in range(4)} == {len(seats) // 4}
    assert rotations("1v3", 6) == [(s,) for s in range(6)]
    with pytest.raises(ValueError):
        rotations("2v2_opposite", 5)


def test_bootstrap_ci_covers_a_known_mean_about_95_percent():
    rng = np.random.default_rng(1)
    trials, true_mean = 300, 1.5
    hits = 0
    for t in range(trials):
        x = rng.normal(true_mean, 4.0, size=200)
        _, lo, hi = bootstrap_ci(x, resamples=1000, seed=t)
        hits += lo <= true_mean <= hi
    assert 0.90 <= hits / trials <= 0.99


def test_sealed_deals_refused_without_the_sealed_run_flag():
    with pytest.raises(SystemExit):
        gate.main(["--set", "sealed", "--n", "10"])
