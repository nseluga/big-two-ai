# Big Two AI — CLAUDE.md

## Key Documents
All of `docs/` is gitignored and local-only (not on GitHub, absent from worktrees — read via `~/big-two-ai/docs/`).
- **Architecture spec:** [`docs/architecture-plan.md`](docs/architecture-plan.md) — goals, design decisions, frozen rules, stages, pre-registered experiments
- **Research manifest:** [`docs/research-manifest.md`](docs/research-manifest.md) — frozen rules, working style, stage→slug table
- **Decision log:** [`docs/decision-log.md`](docs/decision-log.md) — append-only
- **Rules:** [`docs/rules-home.md`](docs/rules-home.md), [`docs/rules-literature.md`](docs/rules-literature.md)
- **Project index:** `~/os/projects/big-two-ai/README.md`

## Project Structure
```
big-two-ai/
├── src/bigtwo/
│   ├── engine/      # variants, bitmask state, legal moves, dealing
│   ├── agents/      # random, greedy, heuristic, dmc, search, external wrappers
│   ├── dmc/         # actors, learner, buffers, nets (control, attention)
│   ├── eval/        # duplicate deals, formats, bootstrap, alpha-rank
│   └── config/      # frozen configs, sealed deal manifests
├── tests/
├── results/         # seeded, config-committed outputs
├── dashboard/       # GitHub Pages (static), ONNX models
├── external/        # pinned bot repos as submodules
└── docs/            # spec, manifest, decision log, rules; lab notebook is gitignored
```

## Stack & Compute
- Python/NumPy bitmask simulator first; port the hot path only after a measured bottleneck. PyTorch for DMC. ONNX for the dashboard.
- Compute: HMC CS GPU boxes (unverified) with a rented-GPU fallback; see plan §9.

## Rules that bind every session
- **Go slow so Nate learns.** Every stage opens with a stage plan (`docs/stage-plans/<slug>.md`) approved before any code. Small steps, explain each before making it, wait for Nate's go-ahead between steps. No `/dev-team-auto`; subagents only on an approved step. Overrides the global "orchestrate, don't ask" default.
- Frozen rules in the manifest. Sealed deal sets and the eval config are never edited in place.
- Any strength claim needs duplicate deals, CIs, and ≥3 seeds.
- No variant-specific hand features; tune on the literature variant only.
- Every stage ends with a learning checkpoint Nate passes before the next stage starts.
- Nothing tracked in git carries a bare stage letter/number — use the manifest's slugs.

## ML Verification Gate
Any session that changes model architecture, loss code, the simulator's rules, or the eval pipeline passes the blocking gates in `~/.claude/skills/ml-engineer/SKILL.md` before a real training run or a recorded result.

## Blocking Questions
A decision not covered by the plan or the decision log: stop and ask. Candidates: new input features, reward changes, a training-technique change, scope expansion.

## Session Startup
1. Read `docs/research-manifest.md`, then plan §6 for the current stage.
2. Check `~/os/projects/big-two-ai/README.md` for status.
3. For research/build sessions, invoke `/research-partner big-two-ai`.
