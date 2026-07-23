# ADR-004: Publish immutable runs with an atomic directory rename

**Status:** Accepted
**Date:** 2026-07-23
**Deciders:** PaperDRM maintainer

## Context

V1 writes evaluation JSON and images into the repository root, then copies a
selected set into `results/<dataset>/`. A failed run can therefore expose a
mixture of new, stale and partially written files. The persistence behaviour is
also coupled to report generation and cleanup, which makes numerical execution
hard to test without filesystem side effects.

V2 already returns one strict, versioned `PipelineResult`. It now needs a
storage boundary that preserves provenance, cannot silently replace history and
never exposes an incomplete run as complete.

## Decision

Add `paperdrm.persistence.RunStore` with this published layout:

```text
runs/<dataset>/<run-id>/
  manifest.json
  result.json
  artifacts/
    diagnostics/
    overlays/
    reports/
```

The store validates safe dataset, run and artifact paths. It serializes strict
JSON, copies artifacts and records their byte size and SHA-256 digest in a
private sibling directory. After syncing the files, it publishes the complete
directory with one same-filesystem rename. A per-run exclusive lock closes the
race between checking and publishing for cooperating writers. Existing runs
and active locks produce `FileExistsError`; no run is overwritten.

Detection and evaluation do not import or call the store. Orchestration chooses
when to persist a successful `PipelineResult`. The manifest is the authoritative
index of persisted artifacts; presentation code is a later consumer.

## Options considered

### Continue copying individual root files

This keeps V1 behaviour but retains stale-file contamination, partial output
visibility and hidden coupling between computation, cleanup and reporting.

### Atomically replace each file

Individual files would be valid, but readers could still observe a manifest,
result and artifact set from different moments.

### Atomically publish one immutable run directory

This adds a small amount of temporary-directory and lock management, but gives
the whole run one visibility boundary and preserves previous runs by default.

## Consequences

- Readers see either no run directory or a complete run directory.
- Strict JSON rejects NaN and infinity instead of producing non-standard data.
- Run identifiers and artifact destinations cannot escape the configured root.
- Artifacts have recorded integrity metadata.
- A crash may leave a hidden lock or temporary directory; it never exposes that
  directory as a completed run. Automated stale-lock recovery is deferred until
  ownership and timeout semantics are defined.
- V1 export and report rendering remain adapters over stored V2 data; they are
  not part of `RunStore`.

## Action items

- [x] Implement atomic run persistence and immutable overwrite semantics.
- [x] Add failure-cleanup, traversal, integrity and strict-JSON tests.
- [ ] Add a backwards-readable V1 export adapter.
- [ ] Make report generation consume the stored policy version.
