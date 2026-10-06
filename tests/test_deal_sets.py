"""Sealed and dev deal sets regenerate to their logged hashes."""

import pytest

from bigtwo.config import deal_sets
from bigtwo.config.deal_sets import DEAL_COUNT, HASHES, SEEDS, load
from bigtwo.engine.deal import CONFIGS


@pytest.mark.parametrize("name", list(SEEDS))
@pytest.mark.parametrize("players, decks", CONFIGS)
def test_regenerated_deals_match_logged_hash(name, players, decks):
    assert len(load(name, players, decks)) == DEAL_COUNT


def test_every_set_has_its_own_hash():
    assert set(HASHES) == {(name, p, d) for name in SEEDS for p, d in CONFIGS}
    assert len(set(HASHES.values())) == len(HASHES)


def test_wrong_hash_is_refused(monkeypatch):
    monkeypatch.setitem(deal_sets.HASHES, ("dev", 4, 1), "0" * 64)
    with pytest.raises(RuntimeError):
        load("dev", 4, 1)


def test_n_out_of_range_is_refused():
    with pytest.raises(ValueError):
        load("dev", 4, 1, n=DEAL_COUNT + 1)
