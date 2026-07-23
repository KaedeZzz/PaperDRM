# ADR-001: Replace V1 incrementally behind frozen contracts

**Status:** Accepted
**Date:** 2026-07-21
**Deciders:** PaperDRM maintainer
**Baseline:** `baseline-v1` (`0c43240`)

## Context

PaperDRM V1 contains working single-image and multi-phi detectors, but its
architecture couples configuration, data loading, detection, plotting,
evaluation, root-directory staging, archiving and report generation in
`main.py`. Result data is spread across several unversioned JSON files. The
repository also mixes core code with a known-biased legacy detector,
one-off experiments, generated results, a report and a logbook.

The nine-folio benchmark includes eight estimates within 10% of manual ground
truth and one important known failure (`Ff4-15_f24r`, approximately -55.5%). A
rewrite must preserve the ability to see that failure; silently improving the
presentation or changing the metric would destroy the comparison oracle.

Constraints:

- research data and manual annotations must not be lost;
- algorithm changes must be distinguishable from architecture changes;
- V1 must remain runnable and comparable throughout migration;
- the CLI and current result artifacts need a compatibility window;
- large real-image runs are too expensive for every unit-test cycle.

## Decision

Use an in-repository strangler migration on `codex/v2-rewrite`.

V1 remains untouched as the reference implementation at `baseline-v1`.
Machine-readable contracts freeze its CLI, accepted configuration fields,
track routing, managed artifacts and minimum JSON key paths. A compact
nine-folio benchmark freezes the primary scientific comparison. V2 components
are introduced behind typed interfaces and must pass compatibility gates
before the corresponding V1 path is retired.

Architecture work and algorithm work are kept in separate commits and review
steps. Phase 0 documents and tests behaviour only; it does not move packages,
delete files or change numerical methods.

## Options considered

### Option A: Big-bang rewrite in a new tree

| Dimension | Assessment |
|---|---|
| Initial simplicity | High |
| Behavioural comparison | Low |
| Data/result migration risk | High |
| Time to first trustworthy result | Long |

Pros: clean structure immediately; no temporary adapters.
Cons: mixes architectural and numerical changes, makes regressions hard to
localise, and encourages reinterpreting known failures after the fact.

### Option B: Incremental replacement behind contracts

| Dimension | Assessment |
|---|---|
| Initial simplicity | Medium |
| Behavioural comparison | High |
| Data/result migration risk | Low |
| Temporary code | Medium |

Pros: each replacement is measurable, V1 stays runnable, and known failures
remain visible.
Cons: compatibility adapters exist temporarily and cleanup must be enforced by
explicit exit gates.

### Option C: Continue patching V1 in place

| Dimension | Assessment |
|---|---|
| Short-term effort | Low |
| Long-term maintainability | Low |
| Scientific traceability | Medium |
| Architectural improvement | Low |

Pros: minimal immediate churn.
Cons: retains orchestration and file-system coupling and makes future result
schema changes increasingly fragile.

## Trade-off analysis

Option B introduces temporary duplication, but the project values scientific
traceability over the shortest code path. Versioned contracts make numerical
changes reviewable: a benchmark difference must be declared as an intentional
algorithm change rather than appearing as a side effect of file movement or
serialization cleanup.

## Consequences

- `main.py` remains a compatibility entry point until the V2 CLI passes parity.
- V1 JSON files remain readable during the compatibility window; V2 may add a
  run manifest and a versioned aggregate result, but cannot silently remove V1
  exports.
- Generated artifacts remain in Git during Phase 0. Stopping tracking is a
  later, separately reviewed cleanup with preservation rules.
- The legacy detector is quarantined, not used as a V2 design template.
- The benchmark initially freezes observed behaviour, including known errors;
  improving it requires an explicit algorithm-change record.
- Phase 1 must add at least one small end-to-end fixture because the current
  suite has no versioned real-image integration case.

## Action items

- [x] Record the V1 CLI, configuration and artifact contract.
- [x] Record the nine-folio manual-GT benchmark.
- [x] Define the target dependency boundaries and file dispositions.
- [x] Add contract tests for accidental V1 drift.
- [x] Add the V2 package skeleton and typed models in Phase 1.
- [ ] Add a small, redistributable end-to-end image fixture in Phase 1.
- [ ] Retire each V1 path only after its compatibility and benchmark gates pass.
