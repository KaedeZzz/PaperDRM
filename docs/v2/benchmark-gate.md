# V2 nine-folio benchmark gate

The V2 benchmark evaluator compares verified immutable runs with the frozen
manual-ground-truth definition in `benchmarks/v1-manual-gt.json`. It does not
read V1 staging JSON, spreadsheet estimates or unverified result files.

## Required inputs

For one explicit run ID, all nine directories must exist:

```text
runs/<folio>/<run-id>/
  manifest.json
  result.json
  artifacts/
```

`load_run` verifies manifest/result identity and every declared artifact before
the evaluator reads the canonical `measurement.lines_per_cm` and stored
confidence policy. A missing or damaged run aborts the whole benchmark; samples
are never silently skipped.

## Gate rules

| Frozen V1 status | V2 result | Gate |
|---|---|---|
| within threshold | absolute manual-GT error ≤10% | pass |
| within threshold | absolute manual-GT error >10% | fail: accuracy regression |
| known failure | absolute manual-GT error ≤10% | pass: improved |
| known failure | error >10%, policy is not `accepted` | pass: failure flagged |
| known failure | error >10%, policy is `accepted` | fail: unsafe known failure |

The special handling of the one frozen known failure prevents Phase 0 from
requiring V2 to reproduce a bad estimate while still forbidding silent
acceptance of the same catastrophic error.

## Command

```bash
python scripts/benchmark_v2.py \
  --runs-root runs \
  --run-id benchmark-001 \
  --output generated-benchmarks/v2-benchmark-001.json
```

The output path is optional and is never overwritten. Exit code `0` means the
gate passed, `1` means complete results failed the gate, and `2` means the
benchmark input was missing or invalid.

## Coverage and remaining gap

Unit/integration tests cover benchmark validation, immutable-run reading,
known-failure containment, unsafe acceptance and missing-run failure. The
remaining end-to-end gap is producing all nine V2 runs from the existing MSI
source files: versioned configurations still contain machine-specific Windows
paths. Input-path mapping must be supplied locally; no additional images are
added to the repository.
