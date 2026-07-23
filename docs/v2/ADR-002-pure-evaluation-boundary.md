# ADR-002: Keep evaluation pure and defer policy and persistence

**Status:** Accepted
**Date:** 2026-07-21
**Deciders:** PaperDRM maintainer

## Context

V1 Stage 5 mixes four concerns in orchestration code: computing metrics,
printing interpretations, plotting figures and writing JSON files. The native
V2 detector backend now returns a versioned result, but connecting it directly
to V1's save/plot functions would recreate the same coupling.

Evaluation metrics are evidence, not automatically a confidence decision.
Split-half agreement can be excellent for a stable alias, negative
self-contrast is a polarity contradiction, and a period-search boundary hit
must reject an otherwise favourable result. Persistence also has different
failure and atomicity requirements from numerical evaluation.

## Decision

Introduce a pure evaluator service. It receives the transient detector output
and in-memory images, calls the unchanged V1 metric kernels, and returns compact
typed evaluation models. It performs no printing, plotting or file writing.

The native backend composes detection and evaluation, then maps both into one
`PipelineResult`. Confidence/rejection policy remains a separate Phase 3
consumer of diagnostic evidence. Run directories, manifests and V1 JSON export
remain a separate Phase 4 persistence concern.

## Options considered

### Option A: Keep evaluation inside the detector backend

| Dimension | Assessment |
|---|---|
| Initial code | Low |
| Testability | Medium |
| Coupling | High |
| Reuse | Low |

Pros: few objects and calls.
Cons: detector changes can alter policy or persistence, and metrics cannot be
re-run independently.

### Option B: Pure evaluator returning typed evidence

| Dimension | Assessment |
|---|---|
| Initial code | Medium |
| Testability | High |
| Coupling | Low |
| Reuse | High |

Pros: deterministic, file-free tests; clear scientific provenance; confidence
and persistence can evolve independently.
Cons: requires explicit mapping from legacy dictionaries into typed models.

### Option C: Preserve the V1 save/plot Stage 5 API

| Dimension | Assessment |
|---|---|
| Compatibility | High |
| Headless execution | Low |
| Atomic persistence | Low |
| Long-term maintainability | Low |

Pros: minimal immediate migration.
Cons: retains root-directory side effects and presentation dependencies in the
core pipeline.

## Trade-off analysis

Option B adds mapping code, but that code becomes the explicit schema boundary.
This is preferable to serializing internal NumPy arrays or allowing report
wording to determine scientific state. Full raw distributions can later be
stored as optional diagnostic artifacts without bloating the primary result.

## Consequences

- Native V2 evaluation can run without creating files or figures.
- `PipelineResult` contains compact interval, fit, contrast, split-half and
  segmented wire-width evidence.
- No confidence label is assigned until Phase 3 defines precedence rules.
- V1 save/plot helpers remain available only through compatibility paths.
- Large diagnostic arrays are transient and are not part of the primary schema.

## Action items

- [x] Add typed evaluation result models.
- [x] Wrap the existing pure metric kernels in an evaluator service.
- [x] Compose evaluation into the native V2 backend.
- [x] Define confidence/rejection precedence in Phase 3.
- [ ] Add atomic run persistence and V1 export adapters in Phase 4.
