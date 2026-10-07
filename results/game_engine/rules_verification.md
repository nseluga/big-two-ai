# Rules verification: game_engine, rules and legal moves

Code: branch `rules-legal-moves`. Home `68a31fd`, literature `6340a7c`. Literature authority: Charlesworth `eea4b04` (submodule `external/big2_PPOalgorithm`).

## Stage-plan runs (§4)

| Check | Command | Result | Wall time |
|---|---|---|---|
| Home invariants, 10k games × 6 configs | `INVARIANT_GAMES=10000 uv run pytest tests/test_invariants.py -k home` | pass | 426.7 s (3p 43.1, 4p 44.1, 5p 40.5, 6p 81.6, 7p 106.7, 8p 110.4) |
| Literature invariants, 10k games 4p/1d | `INVARIANT_GAMES=10000 uv run pytest tests/test_invariants.py -k literature` | pass | 163 s (run alongside the diff run below) |
| Differential vs Charlesworth, 10k lockstep games + 7 crafted states | `DIFF_GAMES=10000 uv run pytest tests/test_charlesworth_diff.py` | 10,007 / 10,007 pass | 444 s |

Differential test: both engines play the same deal in lockstep. At every step the legal-move sets (as card sets) and the seat to act must match, and at the end so must the rewards. Crafted states cover tables random play does not reach: 5,000 random games never put a straight flush on the table with a higher one in the follower's hand.

## Mutation checks (each must make a test fail)

| Mutation | Caught by |
|---|---|
| Home: 6-7 allowed when following; pass lock removed; good-copy tie inverted; 2-deck quad as bomb | home rules tests (all 4) |
| Literature: straight-flush quirk removed | rules tests; crafted diff state (random diff, 5,000 games: **missed**) |
| Literature: pass lock added | rules tests; random diff |
| Literature: straight/flush order swapped | rules tests; random diff |
| Literature: forced 3♦ skipped | rules tests |
| Literature: missing-triple quirk removed | rules tests; random diff; crafted diff |
| Literature: full house keyed by top card, not triple | rules tests; random diff; crafted diff |
| Literature: quad/two-pair order swapped | crafted diff |
| Literature: winner scores 0 | rules tests |

## Incident: stale bytecode (2026-10-06)
The first 10k diff run showed 254 failures. Cause: the quad/two-pair mutation leaves the file the same byte size, and the mutation and its restore landed in the same second, so Python kept the mutated `variants.pyc` as valid. Fix: delete `__pycache__`; both 10k runs above were repeated after the clear. Future mutation runs: set `PYTHONDONTWRITEBYTECODE=1`.

## Review (ml-engineer Mode 4, fresh-context Opus, diff 82a9b5d..6340a7c)
Verdict: **PASS**, no blocking issues.
- 10 planted bugs, each caught.
- His code traced by hand, branch by branch, against the engine.
- `legal_moves` takes 96–297 µs per call, with at most 530 moves in a state.

Changes applied after review:
- The invariant check now rejects the same cards offered twice, even under two pattern names.
- New crafted diff state: a straight on the table, a straight flush with a lower top card in hand.

Carried forward:
- **Diff test compares card sets only.** A wrong pattern type or strength shows up only if a later step diverges.
- **Literature move order is deterministic but unsorted.** RL code must not use list position as an action index.
- **Open question for Nate:** in a 2-deck game, a leader who holds both copies of the lowest card may currently open with the bad copy alone.
