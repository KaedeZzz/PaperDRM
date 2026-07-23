# V1 file disposition for the V2 migration

This is a plan, not a deletion list to execute in Phase 0. `KEEP` means the
asset remains authoritative. `MIGRATE` means move behind a V2 interface while
preserving behaviour. `REWRITE` means preserve the external contract but
replace the implementation. `QUARANTINE` means retain for history or ablation
without importing it into the core package. `STOP TRACKING` applies only after
an archival/preservation check in Phase 5.

## Top-level assets

| Path | Disposition | V2 destination or rule |
|---|---|---|
| `main.py` | REWRITE | Thin compatibility shim, then `src/paperdrm/cli.py` |
| `src/paperdrm/` | KEEP + MIGRATE | Phase 1 moved the package mechanically; migrate internals module-by-module |
| `tests/` | KEEP + EXPAND | Split into unit, contract and integration suites |
| `configs/` | KEEP | Versioned benchmark/user configurations |
| `data/` | KEEP LOCAL | Never commit or delete; accessed only through IO APIs |
| `benchmarks/` | KEEP | Versioned compact baselines and future small fixtures |
| `contracts/` | KEEP | Compatibility contracts with explicit versions |
| `docs/` | KEEP | Architecture, research audit and user documentation |
| `report/` | KEEP SEPARATE | Scholarly output; not imported by the Python package |
| `logbook/` | KEEP SEPARATE | Historical research record; never automated cleanup |
| `results/` | SPLIT | Preserve manual GT and selected baselines; stop tracking generated runs later |
| `scripts/` | SPLIT | Promote maintained tools; quarantine one-off experiments |
| `exp_param.yaml` | KEEP | Compatibility/default example until config migration |
| `README.md` | REWRITE LATER | Update at each completed migration gate |
| `CONVERSATION_LOG.md` | ARCHIVE | Historical project record, outside runtime concerns |
| `requirements.txt` | REWRITE | Derive install/dev groups from `pyproject.toml` |
| `pyproject.toml` | MIGRATE | Configure `src/` packaging, CLI entry point and test tooling |

## Python package

The paths below are relative to `src/paperdrm/`. The Phase 1 layout move did
not change numerical implementations.

| Current path | Disposition | Notes |
|---|---|---|
| `stage0_loader/settings.py` | REWRITE | Validated, versioned config model; preserve accepted V1 fields |
| `stage0_loader/paths.py` | MIGRATE | `io/paths.py`; remove repository-root assumptions |
| `stage0_loader/image_io.py` | MIGRATE | Pure discovery/loading/cache IO |
| `stage0_loader/imagepack.py` | REWRITE | Replace stateful facade with explicit input models |
| `stage0_loader/inference.py` | MIGRATE | Filename/acquisition inference under IO/config boundary |
| `cache_identity.py` | KEEP + MIGRATE | Preserve fingerprint semantics under `io/cache.py` |
| `stage0_drp/` | MIGRATE | DRP array operations; no CLI or result writing |
| `stage3_detect/simple_detector.py` | KEEP + MIGRATE | Active single-image numerical oracle |
| `stage3_detect/multi_phi_detector.py` | KEEP + MIGRATE | Active preferred DRP numerical oracle |
| `stage3_detect/wire_width.py` | MIGRATE | Experimental output, explicitly labelled |
| `detection_diagnostics.py` | KEEP + MIGRATE | Boundary rejection feeds central confidence policy |
| `stage5_evaluation/` | MIGRATE | Pure metrics returning typed results |
| `result_archive.py` | REWRITE | Atomic run store and manifest; V1 export adapter retained |
| `stage4_viz/` | MIGRATE OUTWARD | Optional `reporting/` dependency, never core detection |
| `stage1_features/` | QUARANTINE | Legacy-route support unless a future method justifies revival |
| `stage2_enhance/` | QUARANTINE | Legacy trig-mask support only |
| `stage3_detect/gabor.py` | QUARANTINE | Known half-period bias; ablation/reference only |
| `legacy/` | QUARANTINE | Move to an archive namespace or tagged history in Phase 5 |
| package `__init__.py` files | REWRITE | Export stable public V2 types, not stage internals |

## Scripts

| Group | Files | Disposition |
|---|---|---|
| Maintained data tools | `fetch_dataset.py`, `infer_drp_config.py`, `detect_paper_roi.py`, `gt_builder.py`, `select_bbox.py` | MIGRATE into documented CLI subcommands or `tools/` |
| Maintained evaluation/reporting | `compare_vs_manual_gt.py`, `generate_report.py`, `make_report_figures.py` | MIGRATE to benchmark/reporting interfaces |
| Reproducible benchmark tools | `phantom_synthetic.py`, `legacy_vs_simple_phantom.py`, `compare_detectors.py`, `spreadsheet_comparison.py` | KEEP under `benchmarks/tools/` |
| Figure-only utilities | `plot_for_report.py`, `plot_phase_correction.py`, `plot_polarity_flip.py`, `make_*overlays.py`, `save_comparison.py` | Move under `report/tools/` |
| One-off diagnostics/preparation | `bg_blur*.py`, `batch_setup_configs.py`, `find_paper_dims.py`, `patch_test_1cm.py`, `rgb2grey.py`, `downscale_figures_for_report.py` | QUARANTINE after reproducibility review |

No script is deleted until its inputs, outputs and use in the report have been
checked. A script imported by the runtime (currently `detect_paper_roi.py`) must
be promoted before the old path can be quarantined.

## Results and preservation rules

Always retain:

- every `manual_gt.json` and its provenance;
- the compact V1 benchmark under `benchmarks/`;
- configuration files needed to reproduce a benchmark;
- selected report figures referenced by `report/main.tex`;
- synthetic fixture definitions and expected numeric outputs.

Candidates to stop tracking after migration:

- generated `report_*.html`;
- detector overlays, diagnostic PNGs and plots reproducible from a run;
- repeated per-dataset evaluation JSON once a versioned benchmark/run archive
  exists outside the source tree;
- repository-root staging artifacts.

Before `git rm --cached` is considered, Phase 5 must verify that the file is
generated, reproducible, not a manual annotation, not referenced as the sole
copy by the written report, and preserved in an external or tagged archive.

## Deletion gate

A current V1 implementation can be removed only when all conditions hold:

1. its V2 replacement has unit tests;
2. the relevant contract tests pass;
3. end-to-end fixture output is equivalent within declared tolerances;
4. the nine-folio benchmark has no unexplained change;
5. documentation no longer points to the old path;
6. the removal is isolated in a reviewable cleanup commit.
