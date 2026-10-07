"""Greedy baseline (decision log 2026-10-06).

With control: shed the most cards, lowest top card among those. Following: the
lowest move that beats the table, bombs last. Passes only when it has no play.
Literature legal_moves order is unsorted, so every choice is an explicit sort.
"""

from bigtwo.agents.base import Agent, strength, top_card
from bigtwo.engine.variants import PASS


class GreedyAgent(Agent):
    name = "greedy"

    def act(self, obs, legal):
        plays = [m for m in legal if m != PASS]
        if not plays:
            return PASS
        if obs.last_move is None:
            return min(plays, key=lambda m: (m.kind == "bomb", -m.cards.bit_count(), top_card(m), m.cards))
        return min(plays, key=lambda m: strength(obs.variant, m))
