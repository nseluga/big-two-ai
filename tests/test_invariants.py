"""Seeded random games for every home config and literature 4p/1d, checking engine invariants at every step.

INVARIANT_GAMES sets games per config (default small; the stage-plan run is
INVARIANT_GAMES=10000 uv run pytest tests/test_invariants.py).
"""

import os
import random

import pytest

from bigtwo.engine.cards import to_mask
from bigtwo.engine.deal import CONFIGS, deal_stream
from bigtwo.engine.game import new_game, step
from bigtwo.engine.variants import PASS, HomeVariant, LiteratureVariant

GAMES_PER_CONFIG = int(os.environ.get("INVARIANT_GAMES", "300"))
DEAL_SEED = 20261006
MAX_STEPS = 2000  # far above the longest random game; hitting it means no termination


def check_step(state, moves, dealt: int, played: int) -> None:
    """Cards conserved and in exactly one place; the seat to act can act and its moves are sound."""
    variant, seat = state.variant, state.turn
    assert moves, "seat to act has no legal move"
    assert state.hands[seat] and not state.passed[seat]
    assert len({m.cards for m in moves}) == len(moves), "same cards offered twice"
    assert all(state.hands[seat] & m.cards == m.cards for m in moves)
    if state.last_move is not None:
        assert all(variant.beats(m, state.last_move) for m in moves if m != PASS)
    seen = played
    for held in state.hands:
        assert seen & held == 0, "card in two places"
        seen |= held
    assert seen == dealt, "cards not conserved"


def random_games(variant, players, decks):
    """Yield each finished random game, checking invariants at every step."""
    for game, (hands, _aside) in enumerate(deal_stream(players, decks, DEAL_SEED, GAMES_PER_CONFIG)):
        rng = random.Random(game)
        masks = [to_mask(h, variant.suit_order) for h in hands]
        dealt = sum(masks)
        state, played = new_game(variant, masks), 0
        for _ in range(MAX_STEPS):
            if variant.is_over(state):
                break
            moves = variant.legal_moves(state)
            check_step(state, moves, dealt, played)
            move = rng.choice(moves)
            step(state, move)
            played |= move.cards
        else:
            pytest.fail(f"game {game} did not terminate in {MAX_STEPS} steps")
        yield state


@pytest.mark.parametrize("players,decks", CONFIGS)
def test_home_random_games_keep_invariants(players, decks):
    variant = HomeVariant(decks)
    for state in random_games(variant, players, decks):
        scores = variant.score(state)
        assert sorted(scores, reverse=True) == [2, 1] + [0] * (players - 2)


def test_literature_random_games_keep_invariants():
    variant = LiteratureVariant()
    for state in random_games(variant, 4, 1):
        scores = variant.score(state)
        winner = state.finish_order[0]
        assert sum(scores) == 0 and state.hands[winner] == 0
        assert all(scores[s] == -state.hands[s].bit_count() < 0 for s in range(4) if s != winner)
