"""Engine core: game state, turn order, pass lock, round end, hand end and scoring."""

import pytest

from bigtwo.engine.cards import HOME_SUIT_ORDER, from_names
from bigtwo.engine.game import new_game, step
from bigtwo.engine.variants import PASS, HomeVariant


def hand(names: str) -> int:
    return from_names(names, HOME_SUIT_ORDER, 1)


def start(*hands: str):
    return new_game(HomeVariant(1), [hand(h) for h in hands])


def play(state, names: str) -> None:
    """Step the legal move that plays exactly `names` for the seat to act."""
    cards = hand(names)
    [move] = [m for m in state.variant.legal_moves(state) if m.cards == cards]
    step(state, move)


def pass_turn(state) -> None:
    assert PASS in state.variant.legal_moves(state)
    step(state, PASS)


def test_new_game_starts_with_opening_leader():
    state = start("4S", "3S 9H", "5S")
    assert (state.turn, state.leader, state.last_move) == (1, 1, None)
    assert state.passed == [False] * 3 and state.finish_order == [] and state.history == []


def test_step_removes_cards_and_records_history():
    state = start("3S 9H", "4S 2H", "5S 6D")
    play(state, "3S")
    assert state.hands[0] == hand("9H")
    assert state.last_move.cards == hand("3S") and state.last_player == 0
    assert state.turn == 1 and len(state.history) == 1


def test_step_rejects_cards_not_held():
    state = start("3S 9H", "4S", "5S")
    move = state.variant.legal_moves(state)[0]
    with pytest.raises(ValueError):
        step(state, move._replace(cards=hand("2H")))


def test_copy_is_independent():
    state = start("3S 9H", "4S 2H", "5S 6D")
    snapshot = state.copy()
    play(state, "3S")
    pass_turn(state)
    assert snapshot.hands == [hand("3S 9H"), hand("4S 2H"), hand("5S 6D")]
    assert snapshot.history == [] and snapshot.passed == [False] * 3 and snapshot.last_move is None


def test_round_winner_leads_next_round():
    state = start("3S 9H", "4S 2H", "5S 6D")
    play(state, "3S")
    play(state, "4S")
    pass_turn(state)  # seat 2
    pass_turn(state)  # seat 0
    assert (state.turn, state.leader, state.last_move) == (1, 1, None)
    assert state.passed == [False] * 3
    assert PASS not in state.variant.legal_moves(state)


def test_pass_locks_seat_out_for_the_round():
    # Seat 1 holds a bomb, passes once, and is skipped until the round ends.
    state = start("3S 5C 9H KS", "4S 8S 8C 8D 8H", "4C 6C JH", "4D 7C QH")
    play(state, "3S")
    pass_turn(state)  # seat 1 locked
    play(state, "4C")
    play(state, "4D")
    assert state.turn == 0
    play(state, "5C")
    assert state.turn == 2  # seat 1 skipped
    play(state, "6C")
    play(state, "7C")
    play(state, "9H")
    pass_turn(state)  # seat 2
    pass_turn(state)  # seat 3
    assert (state.turn, state.last_move) == (0, None)  # round over without seat 1
    assert state.passed == [False] * 4


def test_last_card_ends_round_next_seat_leads():
    state = start("3S 3C", "5S 9S", "6S 9C", "7S 9D")
    play(state, "3S 3C")
    for _ in range(3):
        pass_turn(state)
    assert state.finish_order == [0]
    assert (state.turn, state.leader) == (1, 1)


def test_last_card_lead_wraps_to_seat_zero():
    state = start("4S 9S", "5S 9C", "6S 9D", "3S 3C")
    play(state, "3S 3C")
    for _ in range(3):
        pass_turn(state)
    assert state.finish_order == [3]
    assert state.turn == 0


def test_beating_a_player_who_went_out_wins_the_round():
    state = start("3S", "4S 9S", "5S 9C", "6S 9D")
    play(state, "3S")
    play(state, "4S")
    pass_turn(state)
    pass_turn(state)
    assert state.finish_order == [0]
    assert state.turn == 1


def test_hand_ends_when_second_place_decided_and_scores():
    state = start("3S", "4S 9S", "5S", "6S 9D")
    variant = state.variant
    play(state, "3S")
    assert not variant.is_over(state)
    pass_turn(state)
    play(state, "5S")
    assert state.finish_order == [0, 2]
    assert variant.is_over(state)
    assert variant.score(state) == [2, 0, 1, 0]
