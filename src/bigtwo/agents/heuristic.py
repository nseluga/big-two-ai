"""Scripted heuristic: a hand-breakdown planner (decision log 2026-10-06).

Plan: split the own hand into the fewest combos (ties: fewer low orphan singles),
using the variant's own lead patterns. Precedent for the hand-structure features:
Patwa 2026 "Smart" (shed, break penalty, low orphans, save 2s).

Play: go out if possible. With control, lead the weakest non-control planned combo,
or the strongest when two combos are left. Following, play a planned combo that
beats the table; spend control combos only near the end; otherwise pass. When an
opponent has <= 2 cards, follow with the strongest play instead (block); when leading,
the block fires only against a 1-card opponent (lead a multi-card combo, else the top single).
Tuned on literature dev deals only (frozen rule 5).
"""

from bigtwo.agents.base import Agent, lead_moves, strength, top_card
from bigtwo.engine.cards import RANK_NAMES, SLOTS_PER_RANK
from bigtwo.engine.variants import PASS

TWO = RANK_NAMES.index("2")
LOW_SINGLE_TOP_RANK = RANK_NAMES.index("10")  # a single at or below this rank is a "low orphan"
NEAR_OUT = 2  # an opponent at or below this many cards triggers blocking
ENDGAME_COMBOS = 3  # plan size at which control combos may be spent on follows


def rank_of(slot: int) -> int:
    return slot // SLOTS_PER_RANK


def is_control(move) -> bool:
    """Combos that usually win the round: bombs, quads, straight flushes, anything topped by a 2."""
    return move.kind in ("bomb", "quad", "straightflush") or rank_of(top_card(move)) == TWO


def orphan_cost(move) -> int:
    return int(move.kind == "single" and rank_of(top_card(move)) <= LOW_SINGLE_TOP_RANK)


class HeuristicAgent(Agent):
    name = "heuristic"

    def __init__(self):
        self.root, self.variant = 0, None
        self.memo: dict[int, tuple] = {}
        self.by_low: dict[int, list] = {}

    def plan(self, variant, hand: int) -> tuple:
        """Fewest-combo breakdown of hand; ties broken by fewer low orphan singles."""
        if variant is not self.variant or hand != self.root:
            # Re-enumerate per hand: a subset's lead moves are not always a filtered copy of
            # the root's (literature drops the DCS triple only while all four of a rank are held).
            self.root, self.variant, self.memo, self.by_low = hand, variant, {}, {}
            for move in lead_moves(variant, hand):
                low = move.cards & -move.cards
                self.by_low.setdefault(low, []).append(move)
        return self._best(hand)[1]

    def _best(self, mask: int) -> tuple:
        if mask == 0:
            return (0, 0), ()
        if mask in self.memo:
            return self.memo[mask]
        # The mask's lowest card must start some combo, so only combos whose lowest card it is.
        best = None
        for move in self.by_low.get(mask & -mask, ()):
            if move.cards & mask == move.cards:
                (plays, orphans), rest = self._best(mask & ~move.cards)
                cost = (plays + 1, orphans + orphan_cost(move))
                if best is None or cost < best[0]:
                    best = (cost, (move,) + rest)
        self.memo[mask] = best
        return best

    def act(self, obs, legal):
        plays = [m for m in legal if m != PASS]
        if len(legal) == 1 or not plays:
            return legal[0] if legal else PASS
        out = [m for m in plays if m.cards == obs.hand]
        if out:
            return out[0]
        variant = obs.variant
        plan = self.plan(variant, obs.hand)
        planned = {m.cards for m in plan}
        opponents = [n for s, n in enumerate(obs.cards_left) if s != obs.seat and n > 0]
        danger = min(opponents) <= NEAR_OUT
        in_plan = [m for m in plays if m.cards in planned]
        weakest = lambda ms: min(ms, key=lambda m: (is_control(m), top_card(m), -m.cards.bit_count(), m.cards))
        strongest = lambda ms: max(ms, key=lambda m: strength(variant, m))

        if obs.last_move is None:
            if danger and min(opponents) == 1:
                # Do not hand a 1-card opponent a single to beat; if forced, lead the highest single.
                multi = [m for m in (in_plan or plays) if m.cards.bit_count() > 1]
                return weakest(multi) if multi else strongest(plays)
            if not in_plan:  # opening play must include a set card that breaks the plan
                return weakest(plays)
            return strongest(in_plan) if len(plan) <= 2 else weakest(in_plan)

        if danger:
            return strongest(plays)
        cheap = [m for m in in_plan if not is_control(m)]
        if cheap:
            return min(cheap, key=lambda m: strength(variant, m))
        if in_plan and len(plan) <= ENDGAME_COMBOS:
            return min(in_plan, key=lambda m: strength(variant, m))
        return PASS if PASS in legal else min(plays, key=lambda m: strength(variant, m))
