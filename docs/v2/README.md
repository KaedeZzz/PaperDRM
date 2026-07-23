# PaperDRM V2 migration

V2 is an incremental rewrite of the research pipeline. The trusted V1 state is
the annotated Git tag `baseline-v1` (`0c43240`). V1 remains the behavioural
oracle until V2 passes the same contracts and benchmark gates.

## Phase 0: freeze behaviour and design the boundary

Phase 0 changes no detection algorithm and deletes no V1 file. It establishes:

- the accepted migration decision in
  [`ADR-001-strangler-rewrite.md`](ADR-001-strangler-rewrite.md);
- the target package boundaries and staged migration gates;
- a complete disposition policy in
  [`file-disposition.md`](file-disposition.md);
- the machine-readable V1 interface contract in
  [`../../contracts/v1/contract.json`](../../contracts/v1/contract.json);
- the nine-folio manual-ground-truth baseline in
  [`../../benchmarks/v1-manual-gt.json`](../../benchmarks/v1-manual-gt.json);
- contract tests that detect accidental drift before V2 implementation starts.

Phase 0 is complete when the existing regression suite and the new contract
tests pass from a clean checkout.

## Phase 1 status

Phase 1 established the package and type boundaries without changing detection
math:

- the single authoritative package now uses the `src/paperdrm/` layout;
- `paperdrm.cli` owns the frozen V1 argument parser while `main.py` remains the
  compatibility execution entry point;
- `paperdrm.config` provides an immutable normalised pipeline configuration;
- `paperdrm.models` defines the versioned aggregate result, diagnostic evidence
  and future run manifest;
- `paperdrm.pipeline` validates dataset, track and schema identity at the
  backend boundary;
- `paperdrm.compat.v1` preserves V1 route selection and aggregates existing V1
  result directories into the V2 result model.

The result model intentionally records diagnostic evidence without assigning a
confidence label. Confidence/rejection policy remains Phase 3 work. Executing
the numerical pipeline through a native V2 backend and adding a small licensed
end-to-end fixture are the remaining Phase 1 exit items.

## Target architecture

The intended layout is a destination, not a Phase 0 file move:

```text
src/paperdrm/
  cli.py                 command parsing and exit semantics
  config.py              validated user configuration
  models.py              versioned domain/result models
  pipeline.py            orchestration only
  io/                    image discovery, loading, cache identity
  detection/             single-image and multi-phi estimators
  evaluation/            metrics and confidence/rejection policy
  reporting/             serialization and optional presentation
tests/
  contract/              V1/V2 compatibility gates
  integration/           small versioned end-to-end fixtures
  unit/
benchmarks/
  fixtures/
  baselines/
configs/
docs/
```

Dependencies point inward: CLI and reporting depend on the pipeline and domain
models; the pipeline depends on IO, detection and evaluation interfaces; core
detection does not write files or import plotting/reporting code.

## Migration phases and gates

1. **Phase 0 — contract freeze.** Preserve V1 behaviour, record the benchmark,
   and decide every current file's disposition.
2. **Phase 1 — V2 skeleton.** Add the `src/` package, typed configuration and
   result models. Keep `main.py` as a compatibility shim. Gate: CLI and artifact
   contract parity.
3. **Phase 2 — detector migration.** Move single-image and multi-phi detection
   behind pure interfaces. Gate: fixture parity and no benchmark regression.
4. **Phase 3 — evaluation policy.** Centralise confidence, boundary rejection,
   polarity contradiction and measurement provenance. Gate: schema and report
   interpretation tests.
5. **Phase 4 — result storage/reporting.** Replace root staging files with an
   atomic run directory and an explicit manifest. Gate: deterministic archive
   tests and backwards-readable V1 exports.
6. **Phase 5 — cleanup.** Remove compatibility paths, quarantine experiments
   and stop tracking generated output only after all prior gates pass.

## Non-negotiable invariants

- The headline measurement is the global spectral period converted to
  lines/cm; local peak gaps remain diagnostic statistics.
- A period-search boundary hit invalidates the estimate regardless of other
  favourable metrics.
- Negative self-contrast is a polarity contradiction, never positive
  confidence through an absolute value.
- Split-half stability measures repeatability, not correctness.
- Wire width remains experimental and must carry that qualification.
- Manual ground truth is primary for the nine-folio benchmark; spreadsheet
  values are historical secondary references.
- Raw data, manual annotations, the report and the logbook are never removed by
  automated migration or result cleanup.

## Phase 2 status

The first Phase 2 slice is implemented through
`paperdrm.detection.NativeDetectorBackend`:

- typed in-memory inputs distinguish one image from a multi-phi sequence;
- single-image/SIMPLE and multi-phi execution delegate to the unchanged V1
  numerical kernels;
- period-range conversion and auto-direction routing are centralised at the
  backend boundary;
- detector dictionaries are mapped into versioned spacing, grid, diagnostics
  and experimental wire-width models;
- a deterministic synthetic integration fixture exercises both active routes.

The synthetic fixture validates wiring and known-period recovery but does not
replace the planned licensed real-image fixture or the nine-folio benchmark.
Image loading/preprocessing, presentation helpers and result persistence remain
on the V1 orchestration path for later Phase 2/4 slices.

The next slice adds the pure evaluation boundary described by
[`ADR-002-pure-evaluation-boundary.md`](ADR-002-pure-evaluation-boundary.md).
Native V2 runs now compute compact interval, fit, self-contrast, split-half and
segmented wire-width evidence without printing, plotting or writing files.
Persistence remains intentionally deferred; confidence classification is the
separate Phase 3 policy described below.

## Phase 3 status

Confidence and rejection are now handled by the versioned categorical policy
in [`ADR-003-confidence-and-rejection-policy.md`](ADR-003-confidence-and-rejection-policy.md).
Native V2 results carry both a disposition and a confidence level plus stable
reason codes. Hard rejection is limited to invalid period-search boundaries and
strong polarity contradiction. Fit, local-gap and split-half anomalies remain
review warnings until a larger labelled benchmark supports calibration.

## Phase 4 status

Atomic V2 persistence is implemented by `paperdrm.persistence.RunStore` and
specified in
[`ADR-004-atomic-run-persistence.md`](ADR-004-atomic-run-persistence.md).
Complete runs are published under `runs/<dataset>/<run-id>/` with a strict JSON
result, versioned manifest and integrity metadata for copied artifacts. Existing
runs are immutable, path traversal is rejected and failed writes do not expose
a partial final directory.

The one-way compatibility view in
[`ADR-005-v1-export-is-a-derived-view.md`](ADR-005-v1-export-is-a-derived-view.md)
now verifies stored-run identity and artifact integrity before deriving a fresh
flat V1 result directory. V2 remains the only source of truth. The bilingual
HTML report can now read a verified run directly and displays its stored policy
version, disposition and reason code without reclassifying the underlying
metrics. Reports regenerated from immutable runs are written to a new external
directory rather than modifying the run in place.

The application boundary in
[`ADR-006-application-runner-owns-effects.md`](ADR-006-application-runner-owns-effects.md)
now sequences minimum-image loading, native execution, optional artifact
building and atomic storage. Active DRP routes load only the steepest-theta
image for each selected phi; the full DRP cache and root staging workflow remain
available only through the V1 compatibility entry point during migration.
The standard V2 builder renders its overlays and canonical bilingual reports
inside the private artifact workspace before `RunStore` publishes the run.
Report files therefore share the same all-or-nothing publication and manifest
integrity guarantees as every other V2 artifact.
