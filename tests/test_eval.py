"""Eval harness: mirror test, rotations, bootstrap coverage, sealed-deal guard."""

import numpy as np
import pytest

from bigtwo.agents.base import Agent
from bigtwo.agents.greedy import GreedyAgent
from bigtwo.agents.heuristic import HeuristicAgent
from bigtwo.agents.random import RandomAgent
from bigtwo.engine.deal import deal_stream
from bigtwo.engine.variants import HomeVariant, LiteratureVariant
from bigtwo.eval import gate
from bigtwo.engine.cards import to_mask
from bigtwo.eval.duplicate import FORMATS, deal_deltas, play, rotations
from bigtwo.eval.stats import bootstrap_ci


@pytest.mark.parametrize("variant", [LiteratureVariant(), HomeVariant(1)], ids=["literature", "home"])
@pytest.mark.parametrize("agent", [GreedyAgent, HeuristicAgent])
def test_mirror_gives_exactly_zero_on_every_deal(variant, agent):
    for deal in deal_stream(4, 1, 21, 50):
        assert deal_deltas(variant, deal, agent(), agent(), FORMATS) == dict.fromkeys(FORMATS, 0.0)


def test_mirror_heuristic_literature_on_a_plan_cache_regression_deal():
    # seed 99 deal 25 gave Δ = +1.0 when the plan cache outlived the hand (review 2026-10-06).
    deal = list(deal_stream(4, 1, 99, 26))[25]
    assert deal_deltas(LiteratureVariant(), deal, HeuristicAgent(), HeuristicAgent(), FORMATS) == dict.fromkeys(FORMATS, 0.0)


class Spy(Agent):
    """Greedy that records the seats it was asked to act in."""

    def __init__(self):
        self.inner, self.seats = GreedyAgent(), set()

    def act(self, obs, legal):
        self.seats.add(obs.seat)
        return self.inner.act(obs, legal)


@pytest.mark.parametrize("fmt", FORMATS)
def test_agent_a_sits_only_in_the_rotation_seats(fmt):
    deal = next(iter(deal_stream(4, 1, 21, 1)))
    for seats in rotations(fmt, 4):
        spy = Spy()
        hands = [to_mask(ids, LiteratureVariant().suit_order) for ids in deal[0]]
        play(LiteratureVariant(), hands, [spy if s in seats else GreedyAgent() for s in range(4)])
        assert spy.seats <= set(seats) and spy.seats


@pytest.mark.parametrize("variant", [LiteratureVariant(), HomeVariant(1)], ids=["literature", "home"])
def test_delta_sign_greedy_beats_random(variant):
    deals = list(deal_stream(4, 1, 21, 40))
    for fmt in FORMATS:
        fwd = sum(deal_deltas(variant, d, GreedyAgent(), RandomAgent(seed=0), [fmt])[fmt] for d in deals)
        rev = sum(deal_deltas(variant, d, RandomAgent(seed=0), GreedyAgent(), [fmt])[fmt] for d in deals)
        assert fwd > 0 > rev, fmt


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


def test_gate_refuses_fewer_than_two_deals_before_loading():
    with pytest.raises(SystemExit):
        gate.main(["--set", "dev", "--n", "1"])
