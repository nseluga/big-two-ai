"""Rule variants: patterns, legal-move generation, opening, end condition, scoring.

The engine core (game.py) only asks a Variant who opens, what the seat to act may
play, whether passing locks a seat out, when the hand is over and how it scores.

Move design. `kind` names the pattern AND its shape ("pair", "run5", "multi3x2" =
3 ranks of 2 cards each, "bomb"), so "same pattern, same shape" is `kind ==`.
`key` is the slot of the move's top card; a bomb adds BOMB_SIZE_WEIGHT * size, so
size dominates and equal sizes compare by rank. A play beats the table when it has
the same kind and a larger key, or it is a bomb and the table is not.
"""

from itertools import combinations, product
from typing import NamedTuple

from bigtwo.engine.cards import (
    HOME_SUIT_ORDER,
    LITERATURE_SUIT_ORDER,
    RANK_COUNT,
    RANK_NAMES,
    SLOTS_PER_RANK,
    rank_group,
)


class Move(NamedTuple):
    cards: int  # slot mask of the cards played
    kind: str  # pattern and shape, e.g. "single", "run4", "multi3x2", "bomb", "pass"
    key: int  # strength within the kind; higher beats lower


PASS = Move(0, "pass", -1)

N_OF_A_KIND = {1: "single", 2: "pair", 3: "triple", 4: "quad"}
SIZE_OF_KIND = {kind: size for size, kind in N_OF_A_KIND.items()}
BOMB_SIZE_WEIGHT = 128  # larger than any slot (max 103), so bomb size outranks rank
MIN_SEQUENCE_RANKS = 3  # runs and multi-runs need 3 or more consecutive ranks
MAX_COPIES = 8  # copies of one rank in 2 decks
SIX, SEVEN = RANK_NAMES.index("6"), RANK_NAMES.index("7")


class Variant:
    """Interface the engine core calls; each rule variant subclasses it.

    suit_order: slot encoding for this variant's hands; decks: 1 or 2.
    pass_locks: True when a pass sits a seat out for the rest of the round.
    When False, game.step clears the passed flags on every play, so a round ends
    once every other still-playing seat has passed in a row (literature's
    consecutive-pass counter).
    """

    suit_order: str
    decks: int
    pass_locks: bool

    def legal_moves(self, state) -> list[Move]:
        """Every legal move for state.turn; PASS included when passing is allowed."""
        raise NotImplementedError

    def opening(self, hands: list[int]) -> int:
        """Seat that leads the first round."""
        raise NotImplementedError

    def is_over(self, state) -> bool:
        raise NotImplementedError

    def score(self, state) -> list[int]:
        """Points per seat for a finished hand."""
        raise NotImplementedError


def lowest_dealt_pair(hands: list[int]) -> int:
    """Mask of both deck copies of the lowest (rank, suit) held by anyone."""
    held = 0
    for hand in hands:
        held |= hand
    lowest_slot = (held & -held).bit_length() - 1
    return 0b11 << (lowest_slot & ~1)


def rank_cards(hand: int) -> list[list[int]]:
    """Per rank, the single-bit masks of the held cards, weakest first."""
    groups = []
    for rank in range(RANK_COUNT):
        group, base = rank_group(hand, rank), SLOTS_PER_RANK * rank
        groups.append([1 << (base + bit) for bit in range(SLOTS_PER_RANK) if group >> bit & 1])
    return groups


def sequence_kind(rank_count: int, group_size: int) -> str:
    return f"run{rank_count}" if group_size == 1 else f"multi{rank_count}x{group_size}"


def sequence_shape(kind: str) -> tuple[int, int]:
    """(rank count, group size) of a run or multi-run kind."""
    if kind.startswith("run"):
        return int(kind[3:]), 1
    rank_count, group_size = kind[5:].split("x")
    return int(rank_count), int(group_size)


class HomeVariant(Variant):
    """Nate's home rules (docs/rules-home.md)."""

    suit_order = HOME_SUIT_ORDER
    pass_locks = True

    def __init__(self, decks: int):
        if decks not in (1, 2):
            raise ValueError(f"unsupported deck count: {decks}")
        self.decks = decks
        # 1 deck: a quad is the bomb. 2 decks: quads are normal; 5- to 8-of-a-kind bomb.
        self.min_bomb = 4 if decks == 1 else 5

    def beats(self, move: Move, table: Move) -> bool:
        if move.kind == "bomb":
            return table.kind != "bomb" or move.key > table.key
        return move.kind == table.kind and move.key > table.key

    def opening(self, hands: list[int]) -> int:
        # The good copy's holder leads (also covers holding both); else the bad copy's.
        pair = lowest_dealt_pair(hands)
        good_copy = pair & (pair << 1)
        for seat, hand in enumerate(hands):
            if hand & good_copy:
                return seat
        return next(seat for seat, hand in enumerate(hands) if hand & pair)

    def legal_moves(self, state) -> list[Move]:
        seat = state.turn
        if self.pass_locks and state.passed[seat]:
            return []
        hand = state.hands[seat]
        groups = rank_cards(hand)
        table = state.last_move
        if table is None:
            moves = self._lead_moves(groups)
            if not state.history:
                # Opening play must include the leader's copy of the lowest dealt card.
                required = hand & lowest_dealt_pair(state.hands)
                moves = [move for move in moves if move.cards & required]
            return moves
        moves = [move for move in self._shape_moves(groups, table.kind) if self.beats(move, table)]
        if table.kind != "bomb":
            moves += self._of_a_kind(groups, range(self.min_bomb, MAX_COPIES + 1))
        moves.append(PASS)
        return moves

    def is_over(self, state) -> bool:
        return len(state.finish_order) >= 2

    def score(self, state) -> list[int]:
        points = [0] * len(state.hands)
        points[state.finish_order[0]] = 2
        points[state.finish_order[1]] = 1
        return points

    def _move(self, cards: int, kind: str) -> Move:
        top_slot = cards.bit_length() - 1
        if kind == "bomb":
            return Move(cards, kind, BOMB_SIZE_WEIGHT * cards.bit_count() + top_slot)
        return Move(cards, kind, top_slot)

    def _of_a_kind(self, groups, sizes) -> list[Move]:
        """Every choice of `size` same-rank cards, for each size in sizes."""
        moves = []
        for size in sizes:
            kind = "bomb" if size >= self.min_bomb else N_OF_A_KIND[size]
            for group in groups:
                moves += [self._move(sum(combo), kind) for combo in combinations(group, size)]
        return moves

    def _sequence(self, groups, ranks, group_size, kind) -> list[Move]:
        """Every choice of `group_size` cards from each rank in `ranks`."""
        per_rank = [[sum(combo) for combo in combinations(groups[r], group_size)] for r in ranks]
        return [self._move(sum(choice), kind) for choice in product(*per_rank)]

    def _sequences(self, groups, rank_count, group_size) -> list[Move]:
        """All runs/multi-runs of one shape (no wrap: ranks 3 up to 2)."""
        kind = sequence_kind(rank_count, group_size)
        moves = []
        for start in range(RANK_COUNT - rank_count + 1):
            ranks = range(start, start + rank_count)
            if all(len(groups[r]) >= group_size for r in ranks):
                moves += self._sequence(groups, ranks, group_size, kind)
        return moves

    def _lead_moves(self, groups) -> list[Move]:
        moves = self._of_a_kind(groups, range(1, MAX_COPIES + 1))
        for group_size in range(1, MAX_COPIES + 1):
            for rank_count in range(MIN_SEQUENCE_RANKS, RANK_COUNT + 1):
                moves += self._sequences(groups, rank_count, group_size)
            # Lead-only exception: 6-7 plays as the 3-rank run 5-6-7 (same for multi-runs).
            if len(groups[SIX]) >= group_size and len(groups[SEVEN]) >= group_size:
                moves += self._sequence(groups, (SIX, SEVEN), group_size,
                                        sequence_kind(MIN_SEQUENCE_RANKS, group_size))
        return moves

    def _shape_moves(self, groups, kind) -> list[Move]:
        """Candidate follows with the table's kind and shape (strength not yet checked)."""
        if kind == "bomb":
            return self._of_a_kind(groups, range(self.min_bomb, MAX_COPIES + 1))
        if kind in SIZE_OF_KIND:
            return self._of_a_kind(groups, [SIZE_OF_KIND[kind]])
        return self._sequences(groups, *sequence_shape(kind))


# Literature: a follow must have the table's card count; then higher (tier, key) wins.
LITERATURE_TIER = {"single": 0, "pair": 0, "triple": 0, "twopair": 0, "quad": 1,
                   "straight": 0, "flush": 1, "fullhouse": 2, "straightflush": 3}
STRAIGHT_LENGTH = 5
THREE_DIAMONDS = 1  # slot 0 in literature order


def suit_of(card: int) -> int:
    """Suit strength index (0-3) of a single-card 1-deck mask."""
    return (card.bit_length() - 1) >> 1 & 3


class LiteratureVariant(Variant):
    """Charlesworth's rules at eea4b04 (docs/rules-literature.md), quirks included.

    kind is the poker pattern ("single", "twopair", "fullhouse", ...); key is the top
    card's slot (full house: the triple's top slot), matching his comparisons by card id.
    """

    suit_order = LITERATURE_SUIT_ORDER
    decks = 1
    pass_locks = False

    def beats(self, move: Move, table: Move) -> bool:
        # Quirk: nothing follows a straight flush (enumerateOptions.py:48, :93, :127).
        if table.kind == "straightflush" or move.cards.bit_count() != table.cards.bit_count():
            return False
        return (LITERATURE_TIER[move.kind], move.key) > (LITERATURE_TIER[table.kind], table.key)

    def opening(self, hands: list[int]) -> int:
        return next(seat for seat, hand in enumerate(hands) if hand & THREE_DIAMONDS)

    def legal_moves(self, state) -> list[Move]:
        if not state.history:
            # Quirk: the 3♦ holder's first move is forced to the single 3♦ (big2Game.py:68).
            return [Move(THREE_DIAMONDS, "single", 0)]
        groups = rank_cards(state.hands[state.turn])
        table = state.last_move
        if table is None:  # control: any pattern, no pass
            return self._all_moves(groups)
        return [move for move in self._all_moves(groups) if self.beats(move, table)] + [PASS]

    def is_over(self, state) -> bool:
        return len(state.finish_order) >= 1

    def score(self, state) -> list[int]:
        # Winner: cards left in the other hands; each loser: minus own cards (big2Game.py:336).
        points = [-hand.bit_count() for hand in state.hands]
        points[state.finish_order[0]] = -sum(points)
        return points

    @staticmethod
    def _triples(group: list[int]) -> list[int]:
        # Quirk: fillThreeOfAKinds (gameLogic.py:423) never forms ♦♣♠ from a full rank.
        triples = [sum(combo) for combo in combinations(group, 3)]
        if len(group) == 4:
            triples.remove(group[0] | group[1] | group[3])
        return triples

    def _all_moves(self, groups) -> list[Move]:
        top = lambda cards: cards.bit_length() - 1
        pairs = [[sum(c) for c in combinations(g, 2)] for g in groups]
        triples = [self._triples(g) for g in groups]
        moves = [Move(card, "single", top(card)) for g in groups for card in g]
        moves += [Move(p, "pair", top(p)) for ps in pairs for p in ps]
        moves += [Move(t, "triple", top(t)) for ts in triples for t in ts]
        moves += [Move(sum(g), "quad", top(sum(g))) for g in groups if len(g) == 4]
        for low, high in combinations(range(RANK_COUNT), 2):
            moves += [Move(a | b, "twopair", top(b)) for a in pairs[low] for b in pairs[high]]
        for t_rank, p_rank in product(range(RANK_COUNT), repeat=2):
            if t_rank != p_rank:
                moves += [Move(t | p, "fullhouse", top(t)) for t in triples[t_rank]
                          for p in pairs[p_rank]]
        # Straights (5 consecutive ranks, no wrap) and flushes; a straight flush is both.
        fives = set()
        for start in range(RANK_COUNT - STRAIGHT_LENGTH + 1):
            fives.update(sum(c) for c in product(*groups[start:start + STRAIGHT_LENGTH]))
        cards = [card for g in groups for card in g]
        for suit in range(4):
            fives.update(sum(c) for c in combinations([x for x in cards if suit_of(x) == suit], 5))
        for five in fives:
            moves.append(Move(five, self._five_kind(five), top(five)))
        return moves

    @staticmethod
    def _five_kind(cards: int) -> str:
        slots = [bit for bit in range(cards.bit_length()) if cards >> bit & 1]
        ranks = {slot // SLOTS_PER_RANK for slot in slots}
        is_straight = len(ranks) == STRAIGHT_LENGTH and max(ranks) - min(ranks) == STRAIGHT_LENGTH - 1
        is_flush = len({slot >> 1 & 3 for slot in slots}) == 1
        if is_straight:
            return "straightflush" if is_flush else "straight"
        return "flush"
