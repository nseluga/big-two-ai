"""Seeded random home games for every config, checking engine invariants at every step.

INVARIANT_GAMES sets games per config (default small; the stage-plan run is
INVARIANT_GAMES=10000 uv run pytest tests/test_invariants.py).
"""

import os
import random

import pytest

from bigtwo.engine.cards import HOME_SUIT_ORDER, to_mask
from bigtwo.engine.deal import CONFIGS, deal_stream
from bigtwo.engine.game import new_game, step
from bigtwo.engine.variants import PASS, HomeVariant

GAMES_PER_CONFIG = int(os.environ.get("INVARIANT_GAMES", "300"))
DEAL_SEED = 20261006
MAX_STEPS = 2000  # far above the longest random game; hitting it means no termination


def check_step(state, moves, dealt: int, played: int) -> None:
    """Cards conserved and in exactly one place; the seat to act can act and its moves are sound."""
    variant, seat = state.variant, state.turn
    assert moves, "seat to act has no legal move"
    assert state.hands[seat] and not state.passed[seat]
    assert len(set(moves)) == len(moves)
    assert all(state.hands[seat] & m.cards == m.cards for m in moves)
    if state.last_move is not None:
        assert all(variant.beats(m, state.last_move) for m in moves if m != PASS)
    seen = played
    for held in state.hands:
        assert seen & held == 0, "card in two places"
        seen |= held
    assert seen == dealt, "cards not conserved"


@pytest.mark.parametrize("players,decks", CONFIGS)
def test_random_games_keep_invariants(players, decks):
    variant = HomeVariant(decks)
    for game, (hands, _aside) in enumerate(deal_stream(players, decks, DEAL_SEED, GAMES_PER_CONFIG)):
        rng = random.Random(game)
        masks = [to_mask(h, HOME_SUIT_ORDER) for h in hands]
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
        scores = variant.score(state)
        assert sorted(scores, reverse=True) == [2, 1] + [0] * (players - 2)
        assert sum(scores) == 3
