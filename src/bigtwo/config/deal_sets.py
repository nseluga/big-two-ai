"""Sealed and dev deal sets. Deals are regenerated from the seed, never stored;
the logged SHA-256 proves a regeneration matches. Never edit in place: re-seal
under a new decision-log entry (frozen rule 3)."""

import hashlib

from bigtwo.engine.deal import deal_stream

DEAL_COUNT = 50_000
SEEDS = {"sealed": 20261006, "dev": 7}  # fixed 2026-10-06; dev is the only set used for tuning

HASHES = {
    ("sealed", 3, 1): "5443860b1d2130e19773dbeb6e1bf67d5aa4694e8f727ac7932ae2751f2095fd",
    ("sealed", 4, 1): "0f152df8022fa266396aa8a84fc6023e1089d52bde31adce6c40d0e4922572f7",
    ("sealed", 5, 1): "4e263ff63e7c5c7ee582e3b98ac116ce1c4869154f59d8d4a4406e8e376f2c9e",
    ("sealed", 6, 2): "7445650e73a5e425eebcc3e98dbcbefdd805a23b9a74a064a2e7a8fc4914bb50",
    ("sealed", 7, 2): "b8497ab01dda36632b77f78e4e5ecfeaa6cb32214c0b4a008ffcbd83f0cacfd4",
    ("sealed", 8, 2): "5aedaaa42f0662d6ddefe9840cabd8de3c2546dd02c911880e0a4a4eec0e8534",
    ("dev", 3, 1): "52b7ca63caadac033b5a9f6784afb160ebee2a1609802b0e913706d9d07e11d3",
    ("dev", 4, 1): "4d3ec29fc7e43cddfd1bfff47ed98432e7d08008ca9aafcdd6ec6b0056d79e36",
    ("dev", 5, 1): "6b07687fb97af97889e28fd2061221589678b2ebf5d2cb483c2e7c855a68f3cd",
    ("dev", 6, 2): "810630c0f0fe894bec0c45df4447e64164ab431f353079cb26d2eb66741e025b",
    ("dev", 7, 2): "89f66aea825b2f4dd04cf8fe687df43e725715b5033eebd5aa82183a477fd14d",
    ("dev", 8, 2): "f00fca7d98c0fbbcbc82dc04fc7a31584a79fd906db38680c276e4b5d28d1268",
}


def config_seed(base, players, decks):
    # One stream per config, so configs never share shuffles.
    return base * 100 + players * 10 + decks


def deals_hash(deals):
    # Each hand's card ids as bytes, then the aside. Sizes are fixed per config, so no separators.
    h = hashlib.sha256()
    for hands, aside in deals:
        for cards in (*hands, aside):
            h.update(bytes(cards))
    return h.hexdigest()


def load(name, players, decks, n=DEAL_COUNT):
    """First n deals of a set, after checking the whole set against its logged hash."""
    if not 0 < n <= DEAL_COUNT:
        raise ValueError(f"n must be in 1..{DEAL_COUNT}, got {n}")
    deals = list(deal_stream(players, decks, config_seed(SEEDS[name], players, decks), DEAL_COUNT))
    if deals_hash(deals) != HASHES[(name, players, decks)]:
        raise RuntimeError(f"{name} deals for {players}p/{decks}d do not match the logged hash")
    return deals[:n]
