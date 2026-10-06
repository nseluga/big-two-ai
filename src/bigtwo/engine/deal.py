"""Dealing: seeded shuffles of canonical card ids into hands.

Deals are reproducible from a seed alone; sealed test deals must regenerate
bit-identically across Python versions.
"""

import random

# The supported (players, decks) configurations.
CONFIGS = [(3, 1), (4, 1), (5, 1), (6, 2), (7, 2), (8, 2)]


def shuffle(items: list, rng: random.Random) -> None:
    """Fisher-Yates shuffle of `items` in place, drawing only from rng.random()."""
    # Only random() is guaranteed stable across Python versions (shuffle/randrange are not).
    for i in range(len(items) - 1, 0, -1):
        j = int(rng.random() * (i + 1))
        items[i], items[j] = items[j], items[i]


def deal(players: int, decks: int, rng: random.Random) -> tuple[list[list[int]], list[int]]:
    """Deal one game. Returns (hands, aside): sorted canonical-id lists, one hand per seat,
    plus the sorted cards left over when the deck does not divide evenly."""
    if (players, decks) not in CONFIGS:
        raise ValueError(f"unsupported config: {players} players, {decks} decks")
    # Canonical ids carry the deck bit in bit 0, so a 1-deck game uses only the even ids.
    cards = [card_id for card_id in range(104) if card_id % 2 < decks]
    shuffle(cards, rng)
    each = len(cards) // players
    hands = [sorted(cards[k * each:(k + 1) * each]) for k in range(players)]
    aside = sorted(cards[players * each:])
    return hands, aside


def deal_stream(players: int, decks: int, seed: int, count: int):
    """Yield `count` deals from one Random(seed), so a longer stream extends a shorter one."""
    rng = random.Random(seed)
    for _ in range(count):
        yield deal(players, decks, rng)
