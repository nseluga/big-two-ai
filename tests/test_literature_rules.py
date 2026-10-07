"""Literature rules (docs/rules-literature.md, Charlesworth eea4b04): patterns, comparisons, quirks, passing, scoring."""

from bigtwo.engine.cards import LITERATURE_SUIT_ORDER, from_names
from bigtwo.engine.game import GameState, new_game, step
from bigtwo.engine.variants import PASS, LiteratureVariant

V = LiteratureVariant()
FILLER = "2S"  # opponents' hands; never the subject of a test


def hand(names: str) -> int:
    return from_names(names, LITERATURE_SUIT_ORDER, 1)


def lead_moves(names: str):
    """Legal moves with control (round leader, mid-hand)."""
    state = GameState(V, [hand(names)] + [hand(FILLER)] * 3, turn=0, leader=0,
                      passed=[False] * 4, history=[(3, PASS)])
    return V.legal_moves(state)


def table(names: str):
    [move] = [m for m in lead_moves(names) if m.cards == hand(names)]
    return move


def follows(names: str, table_names: str) -> set[int]:
    """Card masks of the plays (not pass) a seat holding `names` may make on `table_names`."""
    state = GameState(V, [hand(names)] + [hand(FILLER)] * 3, turn=0, leader=1,
                      last_move=table(table_names), last_player=1, passed=[False] * 4,
                      history=[(1, table(table_names))])
    moves = V.legal_moves(state)
    assert PASS in moves
    return {m.cards for m in moves if m != PASS}


def masks(*names: str) -> set[int]:
    return {hand(n) for n in names}


def test_suit_order_diamonds_clubs_hearts_spades():
    assert follows("7D 7C 7H 7S", "7C") == masks("7H", "7S")
    assert follows("3C 3H 3S 2D", "3D") == masks("3C", "3H", "3S", "2D")


def test_forced_three_of_diamonds_single_opening():
    hands = [hand("4D 5D"), hand("3C 3D 3H 3S"), hand("6D"), hand("7D")]
    state = new_game(V, hands)
    assert state.turn == 1
    assert V.legal_moves(state) == [table("3D")]
    step(state, V.legal_moves(state)[0])
    assert state.turn == 2 and state.last_move.cards == hand("3D")


def test_same_card_count_only():
    assert follows("4D 4C 5S", "3S") == masks("4D", "4C", "5S")
    assert follows("4D 4C 5S", "3D 3C") == masks("4D 4C")


def test_pairs_and_triples_by_top_card():
    assert follows("9C 9H 10D 10C", "9D 9S") == masks("10D 10C")  # 9H < 9S
    assert follows("8D 8S", "8C 8H") == masks("8D 8S")
    assert follows("6D 6C 6S 4D 4C 4H", "5D 5C 5H") == masks("6D 6C 6S")


def test_two_pair_by_top_card_quads_beat_two_pair():
    two_pair = "4D 4C 9D 9H"
    assert follows("5D 5C 9C 9S", two_pair) == masks("5D 5C 9C 9S")
    assert follows("6D 6C 6H 6S", two_pair) == masks("6D 6C 6H 6S")
    assert follows("5D 5C 8C 8S", two_pair) == set()


def test_quads_by_rank_two_pair_never_beats_quads():
    assert follows("5D 5C 5H 5S", "4D 4C 4H 4S") == masks("5D 5C 5H 5S")
    assert follows("3D 3C 3H 3S", "4D 4C 4H 4S") == set()
    assert follows("AD AC 2D 2C", "4D 4C 4H 4S") == set()


def test_no_five_card_quads_plus_kicker():
    assert not any(m.cards == hand("4D 4C 4H 4S 9D") for m in lead_moves("4D 4C 4H 4S 9D"))


def test_five_card_order_straight_flush_fullhouse_straightflush():
    straight, flush, full_house = "3D 4C 5D 6D 7D", "3C 5C 8C 10C QC", "4D 4C 4H 9D 9C"
    assert table(straight).kind == "straight"
    assert table(flush).kind == "flush"
    assert table(full_house).kind == "fullhouse"
    assert table("8H 9H 10H JH QH").kind == "straightflush"
    assert follows(flush.replace("C", "H"), straight) == masks(flush.replace("C", "H"))
    assert follows("5D 5C 5H JD JC", flush) == masks("5D 5C 5H JD JC")
    assert follows("6S 7S 8S 9S 10S", full_house) == masks("6S 7S 8S 9S 10S")
    assert follows("3H 5H 8H 10H KH", full_house) == set()  # flush never beats a full house
    assert follows("8D 9C 10H JS QD", flush) == set()  # straight never beats a flush


def test_straights_and_flushes_by_top_card():
    assert follows("4C 5C 6C 7H 8D", "4D 5D 6D 7D 8C") == set()  # 8D < 8C
    assert follows("4C 5C 6C 7H 8H", "4D 5D 6D 7D 8C") == masks("4C 5C 6C 7H 8H")
    # Flush compares top card id (rank then suit), not suit first.
    assert follows("3D 5D 8D 10D AD", "3S 5S 8S 10S KS") == masks("3D 5D 8D 10D AD")
    assert follows("3S 5S 8S 10S QS", "3D 5D 8D 10D KD") == set()


def test_full_house_by_triple_rank():
    assert follows("5D 5C 5H 3D 3C", "4D 4C 4H AD AC") == masks("5D 5C 5H 3D 3C")
    assert follows("3H 3S 3C 2D 2C", "4D 4C 4H AD AC") == set()


def test_straight_no_wrap_jqka2_allowed():
    kinds = {m.kind for m in lead_moves("JD QC KD AD 2C")}
    assert "straight" in kinds
    assert not {m.kind for m in lead_moves("QD KC AD 2D 3C")} & {"straight", "straightflush"}


def test_quirk_straight_flush_unbeatable():
    assert follows("9S 10S JS QS KS", "4D 5D 6D 7D 8D") == set()


def test_quirk_missing_triple_from_full_rank():
    triples = {m.cards for m in lead_moves("9D 9C 9H 9S") if m.kind == "triple"}
    assert triples == masks("9D 9C 9H", "9D 9H 9S", "9C 9H 9S")  # never 9D 9C 9S
    fulls = {m.cards for m in lead_moves("9D 9C 9H 9S 4D 4C") if m.kind == "fullhouse"}
    assert hand("9D 9C 9S 4D 4C") not in fulls and len(fulls) == 3
    # A 3-card rank keeps its only triple.
    assert {m.cards for m in lead_moves("9D 9C 9S") if m.kind == "triple"} == masks("9D 9C 9S")


def test_control_has_no_pass_any_pattern():
    moves = lead_moves("3D 3C 4D 5D 6D 7D")
    assert PASS not in moves
    assert {"single", "pair", "straightflush"} <= {m.kind for m in moves}


def test_no_pass_lock_three_passes_give_control():
    hands = [hand("3D 9D"), hand("5D 10D"), hand("6D JD"), hand("7D QD")]
    state = new_game(V, hands)
    step(state, V.legal_moves(state)[0])  # seat 0 forced 3D
    step(state, PASS)  # seat 1
    play = next(m for m in V.legal_moves(state) if m.cards == hand("6D"))
    step(state, play)  # seat 2 plays, resetting the pass count
    assert state.turn == 3
    step(state, PASS)  # seat 3
    assert state.turn == 0
    step(state, PASS)  # seat 0
    assert state.turn == 1 and hand("10D") in {m.cards for m in V.legal_moves(state)}  # seat 1 passed earlier, plays now
    step(state, PASS)  # seat 1: third consecutive pass
    assert state.turn == 2 and state.last_move is None and PASS not in V.legal_moves(state)


def test_ends_at_first_out_zero_sum_score():
    hands = [hand("3D"), hand("5D 10D"), hand("6D JD QD"), hand("7D")]
    state = new_game(V, hands)
    step(state, V.legal_moves(state)[0])
    assert V.is_over(state)
    assert V.score(state) == [6, -2, -3, -1]
