"""Tests for dealing: config shapes, reproducibility, and shuffle uniformity."""

import hashlib
import itertools
import random
from collections import Counter

import pytest

from bigtwo.engine.deal import CONFIGS, deal, deal_stream, shuffle

# (players, decks): (cards per hand, cards set aside)
EXPECTED_SIZES = {
    (3, 1): (17, 1),
    (4, 1): (13, 0),
    (5, 1): (10, 2),
    (6, 2): (17, 2),
    (7, 2): (14, 6),
    (8, 2): (13, 0),
}


def test_configs_are_the_six_supported():
    assert CONFIGS == [(3, 1), (4, 1), (5, 1), (6, 2), (7, 2), (8, 2)]


@pytest.mark.parametrize("players,decks", CONFIGS)
def test_hand_and_aside_sizes_and_coverage(players, decks):
    each, aside_size = EXPECTED_SIZES[(players, decks)]
    hands, aside = deal(players, decks, random.Random(1))
    assert len(hands) == players
    assert all(len(h) == each for h in hands)
    assert len(aside) == aside_size
    everything = [c for h in hands for c in h] + aside
    # No duplicates, nothing missing; 1 deck is the even ids (deck bit 0), 2 decks is 0..103.
    assert sorted(everything) == list(range(0, 104, 2 if decks == 1 else 1))
    assert all(h == sorted(h) for h in hands) and aside == sorted(aside)


def test_unsupported_config_raises():
    with pytest.raises(ValueError):
        deal(2, 1, random.Random(0))
    with pytest.raises(ValueError):
        deal(4, 2, random.Random(0))


def test_same_seed_same_deal_different_seed_different_deal():
    assert list(deal_stream(4, 1, 7, 3)) == list(deal_stream(4, 1, 7, 3))
    assert next(deal_stream(4, 1, 7, 1)) != next(deal_stream(4, 1, 8, 1))


def test_deal_stream_prefix_property():
    long_stream = list(deal_stream(5, 1, 3, 10))
    assert long_stream[:5] == list(deal_stream(5, 1, 3, 5))
    assert len(set(map(repr, long_stream))) == 10  # successive deals differ


def test_shuffle_is_uniform_over_permutations():
    rng = random.Random(123)
    counts = Counter()
    for _ in range(24_000):
        items = [0, 1, 2, 3]
        shuffle(items, rng)
        counts[tuple(items)] += 1
    assert set(counts) == set(itertools.permutations(range(4)))  # all 24 appear (incl. identity)
    assert all(850 <= n <= 1150 for n in counts.values())


def test_pinned_first_deal_seed_0():
    # Literal guard: any change to dealing (shuffle, slicing, seeding) must fail here.
    hands, aside = next(deal_stream(4, 1, seed=0, count=1))
    assert hands[0] == [2, 6, 8, 12, 14, 16, 28, 32, 34, 66, 70, 74, 96]
    assert aside == []


# SHA-256 of repr(first 3 deals, seed 0): catches seat order, aside position, 2-deck card order.
PINNED_DEALS = {
    (3, 1): "41f272f726ed77de5d1baf6ca189476554338528abd200ed3107eed08eac6e2c",
    (4, 1): "86fde4ff64ce597b7d7052be126f8a0a8bf1c808d74d1272b942f988d2049cf3",
    (5, 1): "48840dd60d1b669ed2b60879b8e683bd5fdaba066d59d67d770f307702c94f47",
    (6, 2): "5da67f9b7ce700c6f36da6f36ef99e7a0d3f955ac2ebe2399cfabe127ff00d9b",
    (7, 2): "bba4f74837e1e9c6cacf439e739691b100d818b4ccaea7045d34529010769ed6",
    (8, 2): "43c8743ef2752faa2186c050922965639621585eed0937e34feab003e4b85cb6",
}


@pytest.mark.parametrize("players, decks", CONFIGS)
def test_pinned_deals_every_config(players, decks):
    deals = list(deal_stream(players, decks, seed=0, count=3))
    assert hashlib.sha256(repr(deals).encode()).hexdigest() == PINNED_DEALS[(players, decks)]
