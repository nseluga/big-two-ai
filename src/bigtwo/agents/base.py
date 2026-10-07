"""Agent interface and the public observation agents act on.

Leakage guard: an agent sees an Obs, never the GameState, so it cannot read hidden hands.
"""

from dataclasses import dataclass

from bigtwo.engine.game import GameState
from bigtwo.engine.variants import PASS, LITERATURE_TIER, LiteratureVariant, Move, Variant


@dataclass(frozen=True)
class Obs:
    variant: Variant
    seat: int
    hand: int  # own slot mask; the only private information
    cards_left: tuple[int, ...]  # per seat
    last_move: Move | None  # play to beat; None when this seat has control
    last_player: int | None
    passed: tuple[bool, ...]
    history: tuple[tuple[int, Move], ...]  # public (seat, move) log, passes included


def observe(state: GameState) -> Obs:
    seat = state.turn
    return Obs(state.variant, seat, state.hands[seat], tuple(h.bit_count() for h in state.hands),
               state.last_move, state.last_player, tuple(state.passed), tuple(state.history))


class Agent:
    name = "agent"

    def act(self, obs: Obs, legal: list[Move]) -> Move:
        raise NotImplementedError


def top_card(move: Move) -> int:
    return move.cards.bit_length() - 1


def strength(variant: Variant, move: Move) -> tuple:
    """Order key for comparing plays: bombs last, then the variant's own beat order."""
    if isinstance(variant, LiteratureVariant):
        return (0, LITERATURE_TIER[move.kind], move.key, move.cards)
    return (move.kind == "bomb", move.key, move.cards)


def lead_moves(variant: Variant, hand: int) -> list[Move]:
    """Every move this hand could lead with control, from the hand alone (no hidden info)."""
    probe = GameState(variant, [hand, 0], turn=0, leader=0, passed=[False, False],
                      history=[(1, PASS)])  # non-empty history: not the opening play
    return variant.legal_moves(probe)
