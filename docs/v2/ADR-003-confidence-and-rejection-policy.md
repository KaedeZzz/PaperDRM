# ADR-003: Use categorical confidence with conservative hard rejection

**Status:** Accepted
**Date:** 2026-07-21
**Deciders:** PaperDRM maintainer
**Policy version:** `v1`

## Context

V1 assigns reliability labels inside the HTML report template. That makes the
scientific decision dependent on presentation code and provides no stable
machine-readable reason codes. The available nine-folio benchmark is too small
to calibrate a general composite score. In particular, the catastrophic
`Ff4-15_f24r` alias has suspicious evidence but cannot yet be reliably rejected
without risking false rejection of successful samples.

Some evidence has stronger semantics than other evidence:

- a period peak pinned to the configured search boundary means the search range
  is invalid for interpretation;
- self-contrast `z <= -2` is a strong contradiction of the configured polarity;
- high split-half agreement demonstrates repeatability, not correctness;
- fit disagreement and local-gap disagreement are useful warnings but are not
  calibrated hard-rejection rules.

## Decision

Add a pure, versioned confidence policy with two independent outputs:

1. **Disposition:** `accepted`, `review_required`, `rejected`, or
   `insufficient_evidence`.
2. **Confidence:** `high`, `moderate`, `low`, or `unknown`.

Policy `v1` preserves the existing report thresholds and makes precedence
explicit:

1. Search-boundary hit -> rejected.
2. Otherwise self-contrast `z <= -2` -> rejected for polarity contradiction.
3. Missing self-contrast -> insufficient evidence.
4. `z >= 3` -> accepted/high.
5. `z >= 2` -> accepted/moderate.
6. Otherwise -> review required/low.

Fit disagreement, local-gap disagreement above the existing 15% diagnostic
threshold, and split-half standard deviation at or above the existing 1.5 px
“fair” boundary produce reason-coded warnings. They do not override the primary
disposition in policy `v1`.

## Options considered

### Option A: One weighted confidence score

| Dimension | Assessment |
|---|---|
| Simplicity for users | High |
| Calibration evidence | Low |
| Explainability | Low |
| Overfitting risk | High |

Pros: easy ranking and a single threshold.
Cons: weights would be tuned on nine samples and could hide hard contradictions
inside an average.

### Option B: Ordered categorical policy with reason codes

| Dimension | Assessment |
|---|---|
| Simplicity for users | Medium |
| Calibration evidence | Compatible with current evidence |
| Explainability | High |
| Overfitting risk | Low |

Pros: preserves known semantics, exposes precedence, and can be versioned.
Cons: does not automatically reject every known failure.

### Option C: Expose metrics without any policy

| Dimension | Assessment |
|---|---|
| Scientific caution | High |
| Operational usability | Low |
| Consistency | Low |
| Maintenance | Medium |

Pros: avoids premature thresholds.
Cons: every consumer would reinvent labels and precedence.

## Trade-off analysis

Option B is deliberately conservative. Failing to reject every bad sample is
less misleading than claiming an unvalidated composite score is calibrated.
Reason codes preserve the evidence needed to develop a future policy on a
larger labelled set without changing historical decisions silently.

## Consequences

- Confidence is machine-readable and independent of reports.
- Boundary rejection always wins over favourable contrast or stability.
- Negative contrast is never converted to positive confidence.
- Split-half stability can add a warning but can never prove correctness.
- `Ff4-15_f24r` remains an explicit known limitation of policy `v1`.
- Any future threshold or composite model requires a new policy version and a
  benchmark calibration record.

## Action items

- [x] Add typed disposition, confidence and reason codes.
- [x] Implement pure policy `v1` and precedence tests.
- [x] Attach assessments to native V2 results.
- [x] Make report generation consume the versioned policy in Phase 4.
- [ ] Recalibrate only after expanding the labelled benchmark.
