"""Tests for the card model: slots, ordering, names, and mask operations."""

import pytest

from bigtwo.engine.cards import (
    HOME_SUIT_ORDER,
    LITERATURE_SUIT_ORDER,
    card_name,
    pretty,
    slot,
    to_mask,
)

VARIANTS = [HOME_SUIT_ORDER, LITERATURE_SUIT_ORDER]


def deck_ids(decks: int) -> list[int]:
    """All canonical ids in play: with 1 deck the deck bit is always 0, so only even ids."""
    return [i for i in range(104) if i % 2 < decks]


def id_of(name: str, decks: int = 1) -> int:
    """Canonical id for a name such as '3S' (1 deck) or '2Hg' (2 decks)."""
    return next(i for i in deck_ids(decks) if card_name(i, decks) == name)


def test_literal_suit_orders():
    assert HOME_SUIT_ORDER == "SCDH"
    assert LITERATURE_SUIT_ORDER == "DCHS"


@pytest.mark.parametrize("decks", [1, 2])
def test_lowest_and_highest_cards(decks):
    ids = deck_ids(decks)
    suffix = "b" if decks == 2 else ""  # lowest is the bad copy
    top_suffix = "g" if decks == 2 else ""  # highest is the good copy
    for order, low, high in [(HOME_SUIT_ORDER, "3S", "2H"), (LITERATURE_SUIT_ORDER, "3D", "2S")]:
        by_slot = sorted(ids, key=lambda i: slot(i, order))
        assert card_name(by_slot[0], decks) == low + suffix
        assert card_name(by_slot[-1], decks) == high + top_suffix


@pytest.mark.parametrize("order", VARIANTS)
def test_good_copy_beats_bad_copy(order):
    for rank_and_suit in range(52):
        bad, good = rank_and_suit * 2, rank_and_suit * 2 + 1
        assert slot(good, order) > slot(bad, order)


@pytest.mark.parametrize("order", VARIANTS)
def test_slot_order_is_rank_then_suit_then_deck(order):
    def strength_key(card_id):
        rank_and_suit, deck_bit = divmod(card_id, 2)
        rank, suit_index = divmod(rank_and_suit, 4)
        return rank, order.index("SCDH"[suit_index]), deck_bit

    ids = range(104)
    assert sorted(ids, key=lambda i: slot(i, order)) == sorted(ids, key=strength_key)
    assert sorted(slot(i, order) for i in ids) == list(range(104))  # slots are a permutation


def test_home_slot_equals_canonical_id_but_literature_differs():
    assert all(slot(i, HOME_SUIT_ORDER) == i for i in range(104))
    assert slot(id_of("3D"), LITERATURE_SUIT_ORDER) == 0
    assert slot(id_of("3S"), LITERATURE_SUIT_ORDER) == 6


def test_card_names():
    assert card_name(id_of("3S"), 1) == "3S"
    assert card_name(id_of("10H"), 1) == "10H"
    assert card_name(id_of("2D"), 1) == "2D"
    assert card_name(103, 2) == "2Hg"
    assert card_name(102, 2) == "2Hb"


def test_mask_round_trip_and_ops():
    ids = [id_of(n) for n in ("3S", "10H", "2D", "KC")]
    hand = to_mask(ids, HOME_SUIT_ORDER)
    assert hand.bit_count() == 4
    assert [i for i in range(104) if hand >> slot(i, HOME_SUIT_ORDER) & 1] == sorted(ids)
    move = to_mask([id_of("3S"), id_of("KC")], HOME_SUIT_ORDER)
    assert hand & move == move  # contains
    rest = hand & ~move  # remove
    assert rest.bit_count() == 2
    assert rest & move == 0
    assert to_mask([], HOME_SUIT_ORDER) == 0


def test_two_deck_copies_are_separate_bits():
    bad, good = id_of("5Cb", 2), id_of("5Cg", 2)
    assert to_mask([bad], HOME_SUIT_ORDER) != to_mask([good], HOME_SUIT_ORDER)
    both = to_mask([bad, good], HOME_SUIT_ORDER)
    assert both.bit_count() == 2
    assert (both & ~to_mask([good], HOME_SUIT_ORDER)) == to_mask([bad], HOME_SUIT_ORDER)


def test_pretty_lists_low_to_high():
    ids = [id_of(n) for n in ("2H", "3S", "KC")]
    assert pretty(to_mask(ids, HOME_SUIT_ORDER), HOME_SUIT_ORDER, 1) == "3S KC 2H"
    # Same cards, literature order: rank still decides first, so the list is unchanged.
    assert pretty(to_mask(ids, LITERATURE_SUIT_ORDER), LITERATURE_SUIT_ORDER, 1) == "3S KC 2H"
    # Within a rank, suit order differs: 3S < 3D under home, 3D < 3S under literature.
    pair = [id_of("3S"), id_of("3D")]
    assert pretty(to_mask(pair, HOME_SUIT_ORDER), HOME_SUIT_ORDER, 1) == "3S 3D"
    assert pretty(to_mask(pair, LITERATURE_SUIT_ORDER), LITERATURE_SUIT_ORDER, 1) == "3D 3S"
