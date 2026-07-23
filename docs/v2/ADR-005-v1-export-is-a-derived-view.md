# ADR-005: Treat V1 export as a disposable derived view

**Status:** Accepted
**Date:** 2026-07-23
**Deciders:** PaperDRM maintainer

## Context

Phase 4 makes the aggregate V2 result and run manifest canonical. Existing
reporting and comparison scripts still read several flat V1 JSON files from
`results/<dataset>/`. Requiring every consumer to migrate at once would create
an unnecessary cut-over risk, while storing two authoritative result formats
would permit them to drift.

The compact V2 schema intentionally omits some V1 diagnostic arrays and fitted
curves. A compatibility export can therefore preserve the fields needed by old
consumers, but cannot honestly claim byte-for-byte reconstruction of every V1
artifact.

## Decision

Add a verified stored-run reader and a one-way `V1RunExporter`:

1. The reader validates manifest/result schemas, dataset and track identity,
   contained paths, artifact sizes and SHA-256 digests.
2. The exporter derives report-readable V1 JSON documents from `result.json`.
3. Stored artifacts are copied into the flat export by basename after integrity
   verification. Canonical derived JSON wins over same-named stored artifacts.
4. Export targets must be new directories and are published with a temporary
   sibling rename.
5. Nothing imports data from the V1 view back into a stored V2 run.

## Options considered

### Store complete V1 and V2 results as co-equal records

This offers maximum old-tool fidelity but doubles persistence semantics and
creates an unresolved source-of-truth problem.

### Remove V1 support at the Phase 4 boundary

This yields the smallest implementation but breaks existing reporting and
manual comparison workflows before their replacements are ready.

### Generate a one-way compatibility view

This keeps V2 authoritative, allows incremental consumer migration and makes
the known loss of low-level V1 detail explicit.

## Consequences

- Existing report readers can operate on exported V1-shaped files.
- The view is reproducible from the canonical run and safe to delete.
- Missing V2 fields remain missing; fabricated curves, confidence intervals or
  raw samples are never invented.
- Consumers requiring omitted detail must migrate to the V2 schema or request
  that the canonical result model retain that evidence.
- Direct V2 report rendering remains preferable and is the next migration
  slice.

## Action items

- [x] Add verified stored-run loading.
- [x] Add atomic, non-overwriting V1 compatibility export.
- [x] Make report generation consume the stored V2 policy directly.
- [ ] Migrate comparison tools away from flat V1 files.
