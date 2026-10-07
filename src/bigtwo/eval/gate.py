"""Ladder comparison on a deal set: A vs B, duplicate deals, paired bootstrap CI.

    uv run python -m bigtwo.eval.gate --set dev --n 500 --variant literature

Writes results/eval_harness/<set>_<variant>_<players>p<decks>d_<a>_vs_<b>_n<N>.{json,md}.
Sealed deals (frozen rule 3) need --sealed-run and a clean git tree, so the
recorded commit is the code that ran.
"""

import argparse
import json
import math
import statistics
import subprocess
import time
from pathlib import Path

from bigtwo.agents.greedy import GreedyAgent
from bigtwo.agents.heuristic import HeuristicAgent
from bigtwo.agents.random import RandomAgent
from bigtwo.config import deal_sets
from bigtwo.engine.variants import HomeVariant, LiteratureVariant
from bigtwo.eval.duplicate import FORMATS, deal_deltas
from bigtwo.eval.stats import bootstrap_ci

AGENTS = {"random": lambda: RandomAgent(seed=0), "greedy": GreedyAgent, "heuristic": HeuristicAgent}
EFFECT_SIZE = 1.0  # δ, points per hand (decision log 2026-10-05)
POWER_FACTOR = 2.8  # z(0.975) + z(0.80): detectable δ = 2.8σ/√N
DEFAULT_N = 10_000
RESULTS = Path(__file__).resolve().parents[3] / "results" / "eval_harness"


def git(*args) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, cwd=RESULTS.parents[1]).stdout.strip()


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--set", choices=list(deal_sets.SEEDS), required=True)
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--variant", choices=["literature", "home"], default="literature")
    p.add_argument("--players", type=int, default=4)
    p.add_argument("--decks", type=int, default=1)
    p.add_argument("--a", choices=list(AGENTS), default="heuristic")
    p.add_argument("--b", choices=list(AGENTS), default="greedy")
    p.add_argument("--formats", nargs="+", choices=FORMATS, default=["1v3"])
    p.add_argument("--sealed-run", action="store_true", help="required to play sealed deals")
    args = p.parse_args(argv)

    dirty = bool(git("status", "--porcelain", "--untracked-files=no"))
    if args.set == "sealed" and not (args.sealed_run and not dirty):
        raise SystemExit("sealed deals need --sealed-run and a clean git tree (frozen rule 3)")
    variant = LiteratureVariant() if args.variant == "literature" else HomeVariant(args.decks)
    if args.variant == "literature" and (args.players, args.decks) != (4, 1):
        raise SystemExit("literature variant is 4 players, 1 deck")

    deals = deal_sets.load(args.set, args.players, args.decks, args.n)
    a, b = AGENTS[args.a](), AGENTS[args.b]()
    start = time.time()
    rows = [deal_deltas(variant, deal, a, b, args.formats) for deal in deals]
    seconds = time.time() - start

    summary = {}
    for fmt in args.formats:
        deltas = [r[fmt] for r in rows]
        mean, lo, hi = bootstrap_ci(deltas)
        sigma = statistics.stdev(deltas)
        summary[fmt] = {"mean": mean, "ci95": [lo, hi], "sigma": sigma, "pass": lo > 0,
                        "detectable_delta_at_default_n": POWER_FACTOR * sigma / math.sqrt(DEFAULT_N),
                        "power_n_x2": math.ceil(2 * (POWER_FACTOR * sigma / EFFECT_SIZE) ** 2)}
    record = {"set": args.set, "n": args.n, "variant": args.variant, "players": args.players,
              "decks": args.decks, "a": args.a, "b": args.b,
              "deal_set_hash": deal_sets.HASHES[(args.set, args.players, args.decks)],
              "commit": git("rev-parse", "HEAD"), "dirty_tree": dirty,
              "seconds": round(seconds, 1), "seconds_per_deal": seconds / args.n,
              "bootstrap": {"resamples": 10_000, "seed": 0, "method": "percentile"},
              "results": summary, "deltas": rows}

    RESULTS.mkdir(parents=True, exist_ok=True)
    stem = f"{args.set}_{args.variant}_{args.players}p{args.decks}d_{args.a}_vs_{args.b}_n{args.n}"
    (RESULTS / f"{stem}.json").write_text(json.dumps(record, indent=1) + "\n")
    lines = [f"# {args.a} vs {args.b} — {args.variant} {args.players}p/{args.decks}d, {args.set} deals, N = {args.n}", "",
             f"Commit `{record['commit'][:7]}`{' (dirty tree)' if dirty else ''}; deal-set hash `{record['deal_set_hash'][:16]}…`; "
             f"{record['seconds']} s ({record['seconds_per_deal'] * 1000:.1f} ms/deal).", "",
             "| format | mean Δ (pts/hand) | 95% CI | σ of Δ_d | 2.8σ/√10000 | 2·(2.8σ/δ)² | CI lower > 0 |",
             "|---|---|---|---|---|---|---|"]
    for fmt, s in summary.items():
        lines.append(f"| {fmt} | {s['mean']:+.3f} | [{s['ci95'][0]:+.3f}, {s['ci95'][1]:+.3f}] | {s['sigma']:.3f} | "
                     f"{s['detectable_delta_at_default_n']:.3f} | {s['power_n_x2']} | {s['pass']} |")
    (RESULTS / f"{stem}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
