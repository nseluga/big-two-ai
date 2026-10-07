"""Baseline agents: leakage guard, greedy choices, heuristic plan, reproducibility, legality."""

import pytest

from bigtwo.agents.base import Obs, observe
from bigtwo.agents.greedy import GreedyAgent
from bigtwo.agents.heuristic import HeuristicAgent
from bigtwo.agents.random import RandomAgent
from bigtwo.engine.cards import HOME_SUIT_ORDER, LITERATURE_SUIT_ORDER, from_names, to_mask
from bigtwo.engine.deal import deal_stream
from bigtwo.engine.game import GameState, new_game, step
from bigtwo.engine.variants import PASS, HomeVariant, LiteratureVariant
from bigtwo.eval.duplicate import play

LIT, HOME = LiteratureVariant(), HomeVariant(1)


def lit(names):
    return from_names(names, LITERATURE_SUIT_ORDER, 1)


def home(names):
    return from_names(names, HOME_SUIT_ORDER, 1)


def state_for(variant, hand, table=None, others=None):
    """Seat 0 to act holding `hand`; seat 1 made `table` (None = seat 0 has control)."""
    others = others or [variant_filler(variant)] * 3
    history = [(1, table)] if table else [(3, PASS)]
    return GameState(variant, [hand] + others, turn=0, leader=1 if table else 0, last_move=table,
                     last_player=1 if table else None, passed=[False] * 4, history=history)


def variant_filler(variant):
    return lit("2S") if variant is LIT else home("2H")


def move_of(variant, names, to_hand):
    cards = to_hand(names)
    [m] = [m for m in variant.legal_moves(state_for(variant, cards)) if m.cards == cards]
    return m


def act(agent, state):
    return agent.act(observe(state), state.variant.legal_moves(state))


def test_obs_carries_no_hidden_hand():
    # Two states that differ only in how the opponents' hidden cards are split give the same obs.
    a = state_for(LIT, lit("3D 4D"), others=[lit("5D 6D"), lit("7D 8D"), lit("9D 10D")])
    b = state_for(LIT, lit("3D 4D"), others=[lit("7D 10D"), lit("5D 9D"), lit("6D 8D")])
    assert observe(a) == observe(b)
    assert observe(a).hand == lit("3D 4D")
    assert set(Obs.__dataclass_fields__) == {"variant", "seat", "hand", "cards_left", "last_move",
                                             "last_player", "passed", "history"}


def test_greedy_leads_the_biggest_shed_lowest_top_card():
    move = act(GreedyAgent(), state_for(LIT, lit("3D 3C 4D 5H 6S 7C 2S")))
    assert move.cards.bit_count() == 5 and move.kind == "straight"
    assert move.cards & lit("3D") and move.cards.bit_length() - 1 == lit("7C").bit_length() - 1


def test_greedy_follows_with_the_lowest_beater():
    table = move_of(LIT, "5D", lit)
    assert act(GreedyAgent(), state_for(LIT, lit("4S 6D 9H 2S"), table)).cards == lit("6D")
    pair = move_of(LIT, "5D 5C", lit)
    assert act(GreedyAgent(), state_for(LIT, lit("9D 9H 6C 6S 2S"), pair)).cards == lit("6C 6S")


def test_greedy_plays_bombs_last_and_passes_only_when_forced():
    table = move_of(HOME, "10S", home)
    assert act(GreedyAgent(), state_for(HOME, home("9S 9C 9D 9H KS"), table)).cards == home("KS")
    assert act(GreedyAgent(), state_for(HOME, home("9S 9C 9D 9H 4S"), table)).kind == "bomb"
    assert act(GreedyAgent(), state_for(HOME, home("4S 5S"), table)) == PASS


def test_greedy_does_not_lead_a_bomb_over_a_shorter_play():
    assert act(GreedyAgent(), state_for(HOME, home("9S 9C 9D 9H KS"))).kind != "bomb"


def test_heuristic_plan_partitions_the_hand_in_fewest_combos():
    hand = lit("3D 3C 4D 5H 6S 7C")
    plan = HeuristicAgent().plan(LIT, hand)
    assert len(plan) == 2  # straight + the other 3
    union = 0
    for m in plan:
        assert union & m.cards == 0
        union |= m.cards
    assert union == hand


def test_heuristic_plan_depends_only_on_the_current_hand():
    # Literature drops the DCS triple while all four 5s are held; after 5H is played it is back.
    root, sub = lit("5D 5C 5H 5S 9D 9C JS"), lit("5D 5C 5S 9D 9C JS")
    agent = HeuristicAgent()
    agent.plan(LIT, root)
    assert agent.plan(LIT, sub) == HeuristicAgent().plan(LIT, sub)


def test_heuristic_goes_out_when_it_can():
    table = move_of(LIT, "5D 5C", lit)
    assert act(HeuristicAgent(), state_for(LIT, lit("9D 9H"), table)).cards == lit("9D 9H")


def test_heuristic_blocks_a_one_card_opponent():
    # Opponent at 1 card: with control, do not lead a single when a pair is available.
    s = state_for(LIT, lit("3C 4D 4H 9S"), others=[lit("2S"), lit("5D 6D"), lit("7D 8D")])
    assert act(HeuristicAgent(), s).cards.bit_count() > 1


def test_random_is_reproducible_under_a_seed():
    hands = [to_mask(h, LIT.suit_order) for h in next(deal_stream(4, 1, 5, 1))[0]]
    runs = [play(LIT, hands, [RandomAgent(seed=3)] * 4) for _ in range(2)]
    assert runs[0] == runs[1]


@pytest.mark.parametrize("variant", [LIT, HOME], ids=["literature", "home"])
def test_every_agent_plays_only_legal_moves(variant):
    # play() raises on an illegal move; mixed tables exercise every agent in every seat.
    for k, (hands, _) in enumerate(deal_stream(4, 1, 11, 200)):
        masks = [to_mask(h, variant.suit_order) for h in hands]
        agents = [GreedyAgent(), HeuristicAgent(), RandomAgent(seed=k), HeuristicAgent()]
        rot = k % 4
        play(variant, masks, agents[rot:] + agents[:rot])
