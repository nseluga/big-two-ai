"""Card model: canonical ids, variant-specific slots, and bitmask hands.

A canonical id, (rank*4 + suit)*2 + deck_bit, names a physical card and is the same
in every variant (suit index follows CANONICAL_SUIT_ORDER). A slot, (rank*4 + v)*2 +
deck_bit with v the suit's position in the variant's strength order, is the card's
strength rank and its bit position in a hand mask. Higher slot always means stronger.
"""

CANONICAL_SUIT_ORDER = "SCDH"  # fixed suit indexing for canonical ids
HOME_SUIT_ORDER = "SCDH"  # weakest to strongest: spades < clubs < diamonds < hearts
LITERATURE_SUIT_ORDER = "DCHS"  # diamonds < clubs < hearts < spades

RANK_NAMES = ("3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A", "2")  # weakest first


def slot(card_id: int, suit_order: str) -> int:
    """Convert a canonical card id to its slot under the given variant suit order."""
    rank_and_suit, deck_bit = divmod(card_id, 2)
    rank, suit_index = divmod(rank_and_suit, 4)
    strength_in_suit = suit_order.index(CANONICAL_SUIT_ORDER[suit_index])
    return (rank * 4 + strength_in_suit) * 2 + deck_bit


def to_mask(card_ids, suit_order: str) -> int:
    """Build a hand bitmask (one bit per slot) from canonical card ids."""
    mask = 0
    for card_id in card_ids:
        mask |= 1 << slot(card_id, suit_order)
    return mask


def card_name(card_id: int, decks: int) -> str:
    """Text name such as '3S' or '10H'; with 2 decks add 'g' (good deck) or 'b' (bad deck)."""
    rank_and_suit, deck_bit = divmod(card_id, 2)
    rank, suit_index = divmod(rank_and_suit, 4)
    name = RANK_NAMES[rank] + CANONICAL_SUIT_ORDER[suit_index]
    if decks == 2:
        name += "g" if deck_bit else "b"
    return name


def pretty(mask: int, suit_order: str, decks: int) -> str:
    """Space-separated card names of a hand mask, weakest to strongest."""
    # Invert slot() so each set bit maps back to its canonical id.
    id_of_slot = {slot(card_id, suit_order): card_id for card_id in range(104)}
    slots = [i for i in range(104) if mask >> i & 1]
    return " ".join(card_name(id_of_slot[i], decks) for i in slots)
