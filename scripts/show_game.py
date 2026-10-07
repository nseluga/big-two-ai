"""Print one dev deal's duplicate-deal games move by move, then Δ_d by hand vs the harness.

    uv run python scripts/show_game.py --deal 0 --variant literature [--seat 0] [--a heuristic --b greedy]

--seat N shows only the all-B table and the rotation with A in seat N (default: all 4 rotations).
Dev deals only: this script never loads sealed deals.
"""

import argparse

from bigtwo.agents.base import observe
from bigtwo.config import deal_sets
from bigtwo.engine.cards import pretty, to_mask
from bigtwo.engine.game import new_game, step
from bigtwo.engine.variants import PASS, HomeVariant, LiteratureVariant
from bigtwo.eval.duplicate import deal_deltas
from bigtwo.eval.gate import AGENTS


def show(variant, hands, agents, title):
    print(f"\n=== {title} ===")
    name = lambda m: pretty(m, variant.suit_order, variant.decks)
    for s, h in enumerate(hands):
        print(f"  seat {s} [{agents[s].name:9}] {name(h)}")
    state, n = new_game(variant, hands), 0
    while not variant.is_over(state):
        seat, leading = state.turn, state.last_move is None
        legal = variant.legal_moves(state)
        move = agents[seat].act(observe(state), legal)
        assert move in legal
        step(state, move)
        n += 1
        what = "pass" if move == PASS else f"{move.kind:13} {name(move.cards)}"
        left = state.hands[seat].bit_count()
        print(f"  {n:3} {'LEAD' if leading else '    '} seat {seat} {agents[seat].name:9} {what:32} ({left} left)")
    scores = variant.score(state)
    print(f"  finish order {state.finish_order}  scores {scores}")
    return scores


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--deal", type=int, default=0)
    p.add_argument("--variant", choices=["literature", "home"], default="literature")
    p.add_argument("--seat", type=int)
    p.add_argument("--a", choices=list(AGENTS), default="heuristic")
    p.add_argument("--b", choices=list(AGENTS), default="greedy")
    args = p.parse_args()
    variant = LiteratureVariant() if args.variant == "literature" else HomeVariant(1)
    deal = deal_sets.load("dev", 4, 1, args.deal + 1)[args.deal]
    hands = [to_mask(ids, variant.suit_order) for ids in deal[0]]
    a, b = AGENTS[args.a](), AGENTS[args.b]()

    base = show(variant, hands, [b] * 4, f"baseline: all {args.b}")
    diffs = []
    for s in range(4):
        if args.seat is None or s == args.seat:
            t = show(variant, hands, [a if i == s else b for i in range(4)], f"rotation: {args.a} in seat {s}")
        else:
            t = [None] * 4
        diffs.append(None if t[s] is None else t[s] - base[s])

    print("\n=== Δ by hand (1v3) ===")
    for s, d in enumerate(diffs):
        if d is not None:
            print(f"  seat {s}: {args.a} scored {base[s] + d}, {args.b} scored {base[s]} at the baseline table → {d:+}")
    if None not in diffs:
        print(f"  Δ_d = mean of the 4 = {sum(diffs) / 4:+}")
    print(f"  harness deal_deltas: {deal_deltas(variant, deal, AGENTS[args.a](), AGENTS[args.b]())['1v3']:+}")


if __name__ == "__main__":
    main()
