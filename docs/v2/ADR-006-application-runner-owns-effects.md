# ADR-006: Let an application runner own effect sequencing

**Status:** Accepted
**Date:** 2026-07-23
**Deciders:** PaperDRM maintainer

## Context

The native detector and evaluator are file-free, and `RunStore` can atomically
publish a completed result. They still need an application boundary that loads
the minimum required images, invokes the pipeline, optionally builds
presentation artifacts and persists everything as one run. Keeping this flow
inside `main.py` would preserve the existing coupling to root staging files.

The active SIMPLE and multi-phi detectors require only one steepest-theta image
per selected phi. Building the full four-dimensional DRP cache for these routes
is unnecessary work and introduces cache writes before a run is accepted.

## Decision

Add three explicit application-side interfaces:

1. `FilesystemInputProvider` resolves and preprocesses single images or only
   the selected grazing DRP images. It returns in-memory pipeline input, source
   paths, display images and an effective configuration with crop-adjusted FOV.
2. `ApplicationRunner` sequences input preparation, the injected pipeline,
   optional artifact construction and `RunStore.save`.
3. `ArtifactBuilder` is optional and writes only inside a private temporary
   workspace. `RunStore` copies declared outputs before that workspace is
   removed.

The standard builder produces both detector overlays and package-owned
bilingual V2 reports in that workspace. Reports consume the result's stored
confidence policy and never recompute classification thresholds. A rendering
failure occurs before `RunStore.save`, so no partial run becomes visible.

Legacy execution and the full DRP cache remain behind the V1 compatibility
entry point. The new runner does not change `main.py` routing yet.

## Options considered

### Extend `main.py` with direct `RunStore` calls

This is initially small but keeps computation, plotting, staging cleanup and
persistence in one procedural entry point.

### Make the pipeline load and persist its own files

This reduces application code but breaks the pure boundary and makes numerical
tests dependent on filesystem effects.

### Introduce an application service with injected boundaries

This adds a small set of interfaces while preserving pure core execution,
failure isolation and deterministic tests.

## Consequences

- Native runs no longer require root staging files or the full DRP cache.
- Crop and square-crop changes produce an explicit effective physical scale.
- Input provenance is recorded before atomic persistence.
- Artifact generation can evolve without becoming a detector dependency.
- Automatic ROI detection is intentionally absent; V2 runs require an explicit
  ROI or no crop until that heuristic has its own validated boundary.
- The dedicated V2 CLI publishes standard overlays and bilingual reports as
  integrity-checked artifacts of the same immutable run.

## Action items

- [x] Add filesystem input preparation for native tracks.
- [x] Add the injected application runner and temporary artifact workspace.
- [x] Add a concrete overlay artifact builder.
- [x] Add a dedicated V2 CLI before changing `main.py`.
- [x] Add report generation as a pre-persistence artifact builder.
