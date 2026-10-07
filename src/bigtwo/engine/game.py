"""Game state and the engine core: whose turn, apply a move, round and hand end.

Rules live in the Variant; this module only sequences turns. Hands are slot masks
(one int per seat), so playing a move is `hand & ~move.cards`.
"""

from dataclasses import dataclass, field

from bigtwo.engine.variants import PASS, Move, Variant


@dataclass
class GameState:
    variant: Variant
    hands: list[int]  # slot mask per seat
    turn: int  # seat to act
    leader: int  # seat that led the current round
    last_move: Move | None = None  # play to beat; None when the round leader is to act
    last_player: int | None = None  # seat that made last_move
    passed: list[bool] = field(default_factory=list)  # seats that passed this round
    finish_order: list[int] = field(default_factory=list)  # seats in the order they went out
    history: list[tuple[int, Move]] = field(default_factory=list)  # (seat, move), passes included

    def copy(self) -> "GameState":
        """Independent copy for search; Moves are immutable so lists copy shallowly."""
        return GameState(self.variant, self.hands.copy(), self.turn, self.leader,
                         self.last_move, self.last_player, self.passed.copy(),
                         self.finish_order.copy(), self.history.copy())


def new_game(variant: Variant, hands: list[int]) -> GameState:
    """Start a hand from dealt slot masks; the variant picks the opening leader."""
    leader = variant.opening(hands)
    return GameState(variant, list(hands), leader, leader, passed=[False] * len(hands))


def next_seat(state: GameState, after: int, can_act) -> int:
    """First seat clockwise after `after` (wrapping, `after` itself last) passing can_act."""
    seats = len(state.hands)
    return next(seat for seat in ((after + k) % seats for k in range(1, seats + 1)) if can_act(seat))


def step(state: GameState, move: Move) -> None:
    """Apply the acting seat's move in place. The caller passes a move from legal_moves."""
    variant, seat = state.variant, state.turn
    if move == PASS:
        state.passed[seat] = True
    else:
        if state.hands[seat] & move.cards != move.cards:
            raise ValueError(f"seat {seat} does not hold the cards of {move}")
        state.hands[seat] &= ~move.cards
        state.last_move, state.last_player = move, seat
        if not variant.pass_locks:
            state.passed = [False] * len(state.hands)
        if state.hands[seat] == 0:
            state.finish_order.append(seat)
    state.history.append((seat, move))
    if variant.is_over(state):
        return

    # Round ends when every other seat still holding cards has passed since the last play.
    others = [s for s in range(len(state.hands)) if state.hands[s] and s != state.last_player]
    if all(state.passed[s] for s in others):
        winner = state.last_player
        # A winner who just went out hands the lead to the next seat still holding cards.
        lead = winner if state.hands[winner] else next_seat(state, winner, lambda s: state.hands[s])
        state.turn = state.leader = lead
        state.last_move = state.last_player = None
        state.passed = [False] * len(state.hands)
        return

    locks = variant.pass_locks
    state.turn = next_seat(state, seat,
                           lambda s: state.hands[s] and not (locks and state.passed[s]))
