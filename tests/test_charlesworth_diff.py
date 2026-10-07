"""Differential test: LiteratureVariant vs Charlesworth's own env (external/big2_PPOalgorithm @ eea4b04).

Both engines play the same game in lockstep. At every step the legal-move sets
(as card sets), the seat to act and, at the end, the rewards must be identical.
His env deals and auto-plays 3♦; we mirror his deal. DIFF_GAMES sets the game count.
"""

import os
import random
import sys
from contextlib import chdir
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

from bigtwo.engine.game import new_game, step  # noqa: E402
from bigtwo.engine.variants import PASS, LiteratureVariant  # noqa: E402

REPO = Path(__file__).resolve().parents[1] / "external" / "big2_PPOalgorithm"
GAMES = int(os.environ.get("DIFF_GAMES", "300"))
MAX_STEPS = 2000

if not (REPO / "big2Game.py").exists():
    pytest.skip("submodule not checked out (git submodule update --init)", allow_module_level=True)
sys.path.insert(0, str(REPO))
sys.dont_write_bytecode = True  # keep the pinned submodule clean
with chdir(REPO):  # enumerateOptions loads actionIndices.pkl from the cwd at import
    import big2Game  # noqa: E402
    import enumerateOptions as eo  # noqa: E402

INVERSE = {2: eo.inverseTwoCardIndices, 3: eo.inverseThreeCardIndices,
           4: eo.inverseFourCardIndices, 5: eo.inverseFiveCardIndices}


def mask(card_ids) -> int:
    """His card id (value-1)*4 + suit, suit ♦♣♥♠ = 1..4, is our literature slot // 2 + 1."""
    return sum(1 << (int(card) - 1) * 2 for card in card_ids)


def his_moves(game) -> dict[int, int]:
    """His legal actions as {card mask: action index}; pass is mask 0."""
    hand = game.currentHands[game.playersGo]
    moves = {}
    for index in np.nonzero(game.returnAvailableActions())[0]:
        option, n_cards = eo.getOptionNC(index)
        if n_cards == 0:
            cards = 0
        elif n_cards == 1:
            cards = mask([hand[option]])
        else:
            cards = mask(hand[INVERSE[n_cards][option]])
        moves[cards] = int(index)
    return moves


@pytest.mark.parametrize("seed", range(GAMES))
def test_legal_moves_match_charlesworth(seed):
    np.random.seed(seed)
    rng = random.Random(seed)
    his = big2Game.big2Game()
    opener = his.handsPlayed[1].player - 1  # seats 0-3 are his players 1-4
    hands = [mask(his.currentHands[p]) for p in range(1, 5)]
    hands[opener] |= 1  # his reset already removed the auto-played 3♦
    variant = LiteratureVariant()
    ours = new_game(variant, hands)
    assert ours.turn == opener
    [forced] = variant.legal_moves(ours)
    step(ours, forced)

    for _ in range(MAX_STEPS):
        assert ours.turn == his.playersGo - 1
        theirs = his_moves(his)
        legal = variant.legal_moves(ours)
        assert {m.cards for m in legal} == set(theirs), f"step {len(ours.history)}"
        move = rng.choice(legal)
        step(ours, move)
        reward, done, _ = his.step(theirs[move.cards])
        assert done == variant.is_over(ours)
        if done:
            assert [int(r) for r in reward] == variant.score(ours)
            assert move != PASS
            return
    pytest.fail("no termination")


# Crafted states random play almost never reaches (5000 games never put a straight
# flush on the table with a higher one in the follower's hand).
CRAFTED = [  # (follower's hand, table play or None for control)
    ("9S 10S JS QS KS 3D", "4D 5D 6D 7D 8D"),  # straight flush on table
    ("3H 5H 8H 10H KH 6S 7S 8S 9S 10S 5D 5C 5H JD JC", "4D 4C 4H 9D 9C"),  # full house
    ("3D 4D 5D 6D 7D 3C 5C 8C 10C QC", "4H 6H 8H 10H KH"),  # flush; lower-top SF still beats
    ("AD AC 2D 2C 5D 5C 5H 5S", "4D 4C 4H 4S"),  # quads
    ("AD AC 2D 2C 5D 5C 5H 5S", "4D 4C 9D 9H"),  # two pair
    ("9D 9C 9H 9S 4D 4C 4H 4S 3S", None),  # control with full ranks (missing-triple quirk)
    ("JD QC KD AD 2C 10S", "3D 4C 5D 6D 7D"),  # straight; J-Q-K-A-2 is a straight
]


@pytest.mark.parametrize("held,on_table", CRAFTED)
def test_crafted_states_match_charlesworth(held, on_table):
    from bigtwo.engine.cards import LITERATURE_SUIT_ORDER, from_names
    from bigtwo.engine.game import GameState

    def ids(names):
        cards = from_names(names, LITERATURE_SUIT_ORDER, 1)
        return np.array([slot // 2 + 1 for slot in range(cards.bit_length()) if cards >> slot & 1])

    np.random.seed(0)
    his = big2Game.big2Game()
    his.playersGo, his.currentHands[1] = 1, ids(held)
    variant = LiteratureVariant()
    if on_table is None:
        his.control = 1
        table = None
    else:
        his.control = 0
        his.handsPlayed[his.goIndex - 1] = big2Game.handPlayed(ids(on_table), 4)
        table_cards = mask(ids(on_table))
        lead = GameState(variant, [table_cards, 0, 0, 0], 0, 0, passed=[False] * 4, history=[(3, PASS)])
        [table] = [m for m in variant.legal_moves(lead) if m.cards == table_cards]
    state = GameState(variant, [mask(ids(held)), 1, 1, 1], turn=0, leader=3, last_move=table,
                      last_player=None if table is None else 3, passed=[False] * 4,
                      history=[(3, table or PASS)])
    assert {m.cards for m in variant.legal_moves(state)} == set(his_moves(his))
