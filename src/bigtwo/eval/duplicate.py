"""Duplicate-deal evaluation (stage plan §5, plan §5).

Every deal is played once with B in every seat (the baseline table) and once per
seat rotation with agent A in the rotation's seats and B elsewhere. Same cards and
same opponents, so card luck cancels in the paired difference:
    Δ_d = mean over rotations R of [mean A score over R − mean baseline score over R].
With deterministic agents the mirror (A = B) gives Δ_d = 0 exactly.
"""

from bigtwo.agents.base import observe
from bigtwo.engine.cards import to_mask
from bigtwo.engine.game import new_game, step


def rotations(fmt: str, players: int) -> list[tuple[int, ...]]:
    """Seat sets agent A occupies, one per rotation."""
    if fmt == "1v3":
        return [(s,) for s in range(players)]
    if fmt == "3v1":
        return [tuple(t for t in range(players) if t != s) for s in range(players)]
    if players != 4:
        raise ValueError(f"{fmt} needs 4 players, got {players}")
    if fmt == "2v2_adjacent":
        return [(s, (s + 1) % 4) for s in range(4)]
    if fmt == "2v2_opposite":
        return [(0, 2), (1, 3)]
    raise ValueError(f"unknown format {fmt}")


FORMATS = ("1v3", "2v2_adjacent", "2v2_opposite", "3v1")


def play(variant, hands: list[int], agents) -> list[int]:
    """One hand to the end; returns points per seat. Agents only ever see observe(state)."""
    state = new_game(variant, hands)
    while not variant.is_over(state):
        legal = variant.legal_moves(state)
        move = agents[state.turn].act(observe(state), legal)
        if move not in legal:
            raise ValueError(f"{agents[state.turn].name} played an illegal move {move}")
        step(state, move)
    return variant.score(state)


def deal_deltas(variant, deal, a, b, formats=("1v3",)) -> dict[str, float]:
    """Δ_d per format for one dealt (hands, aside) of canonical card ids."""
    hands = [to_mask(ids, variant.suit_order) for ids in deal[0]]
    players = len(hands)
    baseline = play(variant, hands, [b] * players)
    tables: dict[tuple, list[int]] = {}  # rotations shared across formats are played once
    deltas = {}
    for fmt in formats:
        diffs = []
        for seats in rotations(fmt, players):
            if seats not in tables:
                tables[seats] = play(variant, hands, [a if s in seats else b for s in range(players)])
            diffs.append(sum(tables[seats][s] - baseline[s] for s in seats) / len(seats))
        deltas[fmt] = sum(diffs) / len(diffs)
    return deltas
