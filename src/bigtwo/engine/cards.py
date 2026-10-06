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
RANK_COUNT = len(RANK_NAMES)
SLOTS_PER_RANK = 8  # 4 suits x 2 deck bits; a 1-deck hand uses only the even slots
RANK_GROUP_MASK = (1 << SLOTS_PER_RANK) - 1


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


# Import-time lookup tables for both variants: SLOT[order][card_id] and its inverse.
SLOT = {order: [slot(card_id, order) for card_id in range(104)]
        for order in (HOME_SUIT_ORDER, LITERATURE_SUIT_ORDER)}
ID_OF_SLOT = {order: [0] * 104 for order in SLOT}
for _order, _slots in SLOT.items():
    for _card_id, _slot in enumerate(_slots):
        ID_OF_SLOT[_order][_slot] = _card_id


def rank_group(mask: int, rank: int) -> int:
    """The 8 slot bits of one rank in a mask (bit 2*suit_strength + deck_bit)."""
    return (mask >> (SLOTS_PER_RANK * rank)) & RANK_GROUP_MASK


def pretty(mask: int, suit_order: str, decks: int) -> str:
    """Space-separated card names of a hand mask, weakest to strongest."""
    slots = [i for i in range(104) if mask >> i & 1]
    return " ".join(card_name(ID_OF_SLOT[suit_order][i], decks) for i in slots)


def from_names(names: str, suit_order: str, decks: int) -> int:
    """Hand mask from space-separated card names, the inverse of pretty()."""
    id_of_name = {card_name(i, decks): i for i in range(104) if i % 2 < decks}
    return to_mask((id_of_name[name] for name in names.split()), suit_order)
