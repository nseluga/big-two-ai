"""Home rules, one test per rule line in docs/rules-home.md: patterns, following, bombs, opening, passing."""

import pytest

from bigtwo.engine.cards import HOME_SUIT_ORDER, from_names
from bigtwo.engine.game import GameState, new_game
from bigtwo.engine.variants import PASS, HomeVariant


def hand(names: str, decks: int = 1) -> int:
    return from_names(names, HOME_SUIT_ORDER, decks)


def lead_moves(names: str, decks: int = 1):
    """Legal moves for a round leader holding `names`, mid-hand (history non-empty, so not the opening)."""
    filler = hand("2H", decks) if decks == 1 else hand("2Hg", decks)
    state = GameState(HomeVariant(decks), [hand(names, decks), filler, filler], turn=0, leader=0,
                      passed=[False] * 3, history=[(2, PASS)])
    return HomeVariant(decks).legal_moves(state)


def table(names: str, decks: int = 1, kind: str | None = None):
    """The lead move that plays exactly `names` (and has `kind`, when given)."""
    cards = hand(names, decks)
    [move] = [m for m in lead_moves(names, decks) if m.cards == cards and kind in (None, m.kind)]
    return move


def follow_moves(names: str, table_move, decks: int = 1):
    """Legal moves for a seat holding `names` facing `table_move`."""
    filler = hand("2H", decks) if decks == 1 else hand("2Hg", decks)
    state = GameState(HomeVariant(decks), [hand(names, decks), filler, filler], turn=0, leader=1,
                      last_move=table_move, last_player=1, passed=[False] * 3,
                      history=[(1, table_move)])
    return HomeVariant(decks).legal_moves(state)


def plays(moves, decks: int = 1) -> set[int]:
    """Card masks of the non-pass moves."""
    return {m.cards for m in moves if m != PASS}


def masks(*names: str, decks: int = 1) -> set[int]:
    return {hand(n, decks) for n in names}


# Cards and ranking

def test_rank_order_3_low_2_high():
    moves = follow_moves("3H QH KC AS 2S", table("KS"))
    assert plays(moves) == masks("KC", "AS", "2S")


def test_suit_order_spades_clubs_diamonds_hearts():
    assert plays(follow_moves("7S 7D 7H", table("7C"))) == masks("7D", "7H")
    assert plays(follow_moves("3C 3D 3H", table("3S"))) == masks("3C", "3D", "3H")


def test_one_or_two_decks_only():
    with pytest.raises(ValueError):
        HomeVariant(3)


# Opening

def test_lowest_dealt_card_leads_and_opening_includes_it():
    hands = [hand("4S 9H"), hand("5C 5D"), hand("3S 3H 4C 5S 6S"), hand("2H")]
    variant = HomeVariant(1)
    state = new_game(variant, hands)
    assert state.turn == state.leader == 2
    moves = variant.legal_moves(state)
    assert all(m.cards & hand("3S") for m in moves)
    # Opening may be any legal pattern that includes the card.
    assert {"single", "pair", "run3", "run4"} <= {m.kind for m in moves}
    assert hand("3H") not in plays(moves)


def test_three_spades_aside_three_clubs_leads():
    variant = HomeVariant(1)
    hands = [hand("3D 9S"), hand("3C 4S"), hand("5S")]  # 3S set aside
    state = new_game(variant, hands)
    assert state.turn == 1
    assert all(m.cards & hand("3C") for m in variant.legal_moves(state))
    # 3S and 3C both aside: 3D leads.
    assert HomeVariant(1).opening([hand("4S"), hand("3H"), hand("3D")]) == 2
    # All 3s aside: the order continues to 4S.
    assert HomeVariant(1).opening([hand("4C"), hand("4S"), hand("5S")]) == 1


def test_two_decks_good_copy_holder_leads():
    variant = HomeVariant(2)
    hands = [hand("3Sb 9Hg", 2), hand("3Sg 4Cb", 2), hand("5Sb", 2)]
    state = new_game(variant, hands)
    assert state.turn == 1
    assert all(m.cards & hand("3Sg", 2) for m in variant.legal_moves(state))


def test_two_decks_one_player_holds_both_copies_leads():
    variant = HomeVariant(2)
    hands = [hand("4Sb", 2), hand("9Hg", 2), hand("3Sb 3Sg 7Cb", 2)]
    state = new_game(variant, hands)
    assert state.turn == 2
    moves = variant.legal_moves(state)
    assert all(m.cards & hand("3Sb 3Sg", 2) for m in moves)
    assert masks("3Sb", "3Sg", "3Sb 3Sg", decks=2) <= plays(moves)
    assert hand("7Cb", 2) not in plays(moves)


def test_two_decks_only_bad_copy_dealt_its_holder_leads():
    assert HomeVariant(2).opening([hand("5Sg", 2), hand("4Cg", 2), hand("3Sb 2Hg", 2)]) == 2


# Legal patterns

def test_every_pattern_recognized():
    kinds = {m.kind for m in lead_moves("3S 3C 3D 3H 4S 4C 5S 5C 6S 9H")}
    assert {"single", "pair", "triple", "bomb", "run3", "run4", "multi3x2"} <= kinds
    two_deck = {m.kind for m in lead_moves("8Sb 8Cb 8Db 8Hb", 2)}
    assert two_deck == {"single", "pair", "triple", "quad"}


def test_no_full_houses_or_flushes():
    assert {m.kind for m in lead_moves("3S 3C 4S 4C 4D")} == {"single", "pair", "triple"}
    assert {m.kind for m in lead_moves("3H 5H 7H 9H JH")} == {"single"}


def test_run_k_a_2_legal_but_no_wrap():
    moves = plays(lead_moves("QS KS AS 2S 3S 4S"))
    assert masks("KS AS 2S", "QS KS AS", "QS KS AS 2S") <= moves
    assert not masks("AS 2S 3S", "2S 3S 4S", "KS AS 2S 3S") & moves


def test_run_is_any_cards_three_or_more_ranks():
    moves = lead_moves("3S 4H 5D 6C")
    assert masks("3S 4H 5D", "4H 5D 6C", "3S 4H 5D 6C") <= plays(moves)


def test_two_rank_runs_illegal_except_6_7():
    assert {m.kind for m in lead_moves("3S 4S")} == {"single"}
    assert {m.kind for m in lead_moves("9S 10S")} == {"single"}


def test_multi_run_any_group_size():
    moves = lead_moves("3S 3C 3D 3H 4S 4C 4D 4H 5S 5C 5D 5H")
    by_cards = {m.cards: m.kind for m in moves}
    assert by_cards[hand("3S 3C 3D 3H 4S 4C 4D 4H 5S 5C 5D 5H")] == "multi3x4"
    assert by_cards[hand("3S 3C 3D 4S 4C 4D 5S 5C 5D")] == "multi3x3"
    assert by_cards[hand("3D 3H 4D 4H 5D 5H")] == "multi3x2"


def test_two_rank_multi_run_illegal():
    assert {m.kind for m in lead_moves("3S 3C 4S 4C")} == {"single", "pair"}


def test_multi_run_followers_match_rank_count_and_group_size():
    moves = follow_moves("6S 6C 7S 7C 8S 8C 9S 9C", table("3S 3C 4S 4C 5S 5C"))
    assert plays(moves) == masks("6S 6C 7S 7C 8S 8C", "7S 7C 8S 8C 9S 9C")
    triples = follow_moves("7S 7C 7D 8S 8C 8D 9S 9C 9D 10S 10C",
                           table("4S 4C 4D 5S 5C 5D 6S 6C 6D"))
    assert {m.kind for m in triples} == {"multi3x3", "pass"}


def test_6_7_leads_as_5_6_7():
    [move] = [m for m in lead_moves("6S 7C 9H") if m.cards == hand("6S 7C")]
    assert move.kind == "run3"
    assert move.key == hand("7C").bit_length() - 1  # strength = the 7


def test_6_7_multi_run_leads_as_3_rank_multi_run():
    assert table("6S 6C 7S 7C").kind == "multi3x2"
    assert table("6S 6C 6D 7S 7C 7D").kind == "multi3x3"


def test_6_7_legal_in_opening_when_it_holds_the_lowest_card():
    variant = HomeVariant(1)
    state = new_game(variant, [hand("6S 7C"), hand("8S"), hand("9S")])
    assert hand("6S 7C") in plays(variant.legal_moves(state))


def test_6_7_illegal_when_following():
    assert follow_moves("6D 7C", table("3S 4S 5S")) == [PASS]
    assert follow_moves("6H 7H", table("6S 7S")) == [PASS]
    assert follow_moves("6D 6H 7D 7H", table("3S 3C 4S 4C 5S 5C")) == [PASS]


def test_3_run_with_higher_top_beats_6_7():
    six_seven = table("6D 7S")
    assert plays(follow_moves("5S 6S 7C", six_seven)) == masks("5S 6S 7C")
    assert plays(follow_moves("4H 5H 6H", six_seven)) == set()
    multi = table("6D 6H 7S 7C")
    assert plays(follow_moves("5S 5C 6S 6C 7D 7H", multi)) == masks("5S 5C 6S 6C 7D 7H")


# Following

def test_follow_must_match_pattern_and_card_count():
    moves = follow_moves("4C 5C 6C 7C 9D 9H", table("3S 4S 5S"))
    assert {m.kind for m in moves} == {"run3", "pass"}
    assert {m.kind for m in follow_moves("9D 9H 10S", table("3S 3C"))} == {"pair", "pass"}


def test_runs_ranked_by_highest_card():
    # 3S 4S 5C beats 3H 4H 5S: lower cards weaker, top card stronger.
    assert hand("3S 4S 5C") in plays(follow_moves("3S 4S 5C", table("3H 4H 5S")))
    assert plays(follow_moves("3H 4H 5S", table("3S 4S 5C"))) == set()


def test_one_deck_quad_bomb_beats_anything():
    quad = "8S 8C 8D 8H"
    for led in ("2H", "10S JS QS", "2D 2H", "3S 3C 3D", "3S 3C 4S 4C 5S 5C"):
        assert hand(quad) in plays(follow_moves(quad, table(led)))
    assert table(quad).kind == "bomb"


def test_one_deck_only_a_higher_quad_beats_a_quad():
    moves = follow_moves("7S 7C 7D 7H 9S 9C 9D 9H 2S", table("8S 8C 8D 8H"))
    assert plays(moves) == masks("9S 9C 9D 9H")


def test_two_decks_quad_is_a_normal_pattern():
    quad = "8Sb 8Cb 8Db 8Hb"
    assert table(quad, 2).kind == "quad"
    assert follow_moves(quad, table("2Hg", 2), 2) == [PASS]
    moves = follow_moves("7Sb 7Cb 7Db 7Hb 9Sb 9Cb 9Db 9Hb", table(quad, 2), 2)
    assert plays(moves) == masks("9Sb 9Cb 9Db 9Hb", decks=2)
    # Same rank: compared by top card slot.
    assert plays(follow_moves("8Sg 8Cg 8Dg 8Hg", table(quad, 2), 2)) == masks("8Sg 8Cg 8Dg 8Hg", decks=2)


def test_two_decks_bombs_are_5_to_8_of_a_kind():
    eights = "8Sb 8Cb 8Db 8Hb 8Sg 8Cg 8Dg 8Hg"
    bombs = {m.cards.bit_count() for m in lead_moves(eights, 2) if m.kind == "bomb"}
    assert bombs == {5, 6, 7, 8}


def test_two_decks_five_of_a_kind_beats_quad_and_anything():
    five = "3Sb 3Cb 3Db 3Hb 3Sg"
    for led in ("2Hg", "AHb AHg", "8Sb 8Cb 8Db 8Hb", "10Sb JSb QSb KSb"):
        assert hand(five, 2) in plays(follow_moves(five, table(led, 2), 2))


def test_two_decks_bigger_bomb_beats_smaller_and_same_size_by_rank():
    kings = table("KSb KCb KDb KHb KSg", 2)
    six_threes = "3Sb 3Cb 3Db 3Hb 3Sg 3Cg"
    assert hand(six_threes, 2) in plays(follow_moves(six_threes, kings, 2))
    five_aces, five_queens = "ASb ACb ADb AHb ASg", "QSb QCb QDb QHb QSg"
    assert plays(follow_moves(five_aces + " " + five_queens, kings, 2)) == masks(five_aces, decks=2)
    six = table(six_threes, 2)
    assert plays(follow_moves("ASb ACb ADb AHb ASg ACg ADg", six, 2)) >= masks("ASb ACb ADb AHb ASg ACg ADg", decks=2)
    assert hand(five_aces, 2) not in plays(follow_moves(five_aces, six, 2))


def test_good_copy_beats_bad_copy():
    assert plays(follow_moves("2Hg", table("2Hb", 2), 2)) == masks("2Hg", decks=2)
    assert follow_moves("2Hb", table("2Hg", 2), 2) == [PASS]
    assert plays(follow_moves("9Sb 9Hg", table("9Sg 9Hb", 2), 2)) == masks("9Sb 9Hg", decks=2)


# Passing

def test_round_leader_cannot_pass():
    assert PASS not in lead_moves("3S 4S 5S")
    variant = HomeVariant(1)
    assert PASS not in variant.legal_moves(new_game(variant, [hand("3S"), hand("4S"), hand("5S")]))


def test_pass_is_voluntary_with_a_legal_play():
    moves = follow_moves("2H", table("3S"))
    assert hand("2H") in plays(moves) and PASS in moves


def test_passed_seat_gets_no_moves_not_even_a_bomb():
    variant = HomeVariant(1)
    led = table("3S")
    state = GameState(variant, [hand("8S 8C 8D 8H 2H"), hand("4S"), hand("5S")], turn=0, leader=1,
                      last_move=led, last_player=1, passed=[True, False, False], history=[(1, led)])
    assert variant.legal_moves(state) == []
