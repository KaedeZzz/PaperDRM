"""
Collect full-image pipeline measurements plus per-folio manual and spreadsheet
reference values for all 9 folios.

The primary error metric uses ``results/<serial>/manual_gt.json`` when present.
Spreadsheet values are retained as secondary historical references because some
refer to different folios or contain uncertain counts.
"""
import sys, json, yaml
sys.path.insert(0, ".")
from pathlib import Path

DATASETS = [
    # (serial,         spreadsheet_lpc, spreadsheet_note)
    ("Kk1-5_f5v",    9.0,  "manual count on f5v/f9v; spreadsheet stock 235a/b records 12 (different folios)"),
    ("Kk1-5_f9v",    9.0,  "manual count on f5v/f9v; spreadsheet stock 235a/b records 12 (different folios)"),
    ("Hh2-12_f190",  10.0, "spreadsheet stock 0136b locus 176r"),
    ("Ee5-22_f328r", 10.0, "spreadsheet stock 0061a locus 324-325"),
    ("Ff2-6_f140r",  11.0, "spreadsheet stock 0069a locus 138r"),
    ("Ff4-9_f42r",    6.0, "spreadsheet stock 0070b locus 1r"),
    ("Ff4-15_f24r",  13.5, "spreadsheet stock 0076b locus 23r; value '13 (14? hard to tell)'"),
    ("Hh2-10_f24r",  13.5, "spreadsheet stock 0095a locus 1r; value '13 (14)'"),
    ("Ii3-8_f135v",   9.0, "spreadsheet stock 0155b locus 133r"),
]


def load_pipeline(serial: str) -> dict:
    base = Path("results") / serial
    iv  = json.loads((base / "interval_distribution.json").read_text(encoding="utf-8"))
    ww  = json.loads((base / "wire_width_stats.json").read_text(encoding="utf-8"))
    sc  = json.loads((base / "self_contrast.json").read_text(encoding="utf-8"))
    fq  = json.loads((base / "fit_quality.json").read_text(encoding="utf-8"))
    sh  = json.loads((base / "split_half_stability.json").read_text(encoding="utf-8"))

    phys    = iv.get("physical", {})
    spectral_interval_cm = phys.get("spectral_interval_cm")
    if spectral_interval_cm is None:
        period_px = iv.get("period_px_used")
        cm_per_px = phys.get("cm_per_px")
        if period_px is not None and cm_per_px is not None:
            spectral_interval_cm = period_px * cm_per_px
        else:
            spectral_interval_cm = phys.get("mean_interval_cm")
    spectral_lpc = phys.get("spectral_lines_per_cm")
    if spectral_lpc is None and spectral_interval_cm:
        spectral_lpc = 1.0 / spectral_interval_cm
    ww_ph   = ww.get("physical", {})
    fwhm_d  = ww_ph.get("fwhm_mm", {})
    ci      = fwhm_d.get("ci_t", [None, None])

    z       = sc.get("contrast_z", sc.get("z_score"))
    rel_pct = sc.get("contrast_rel", 0) * 100

    sh_mean = sh.get("period_cm_mean")
    sh_std  = sh.get("period_cm_std")
    sh_lpc  = 1.0 / sh_mean if sh_mean else None

    cfg = yaml.safe_load(
        (Path("configs") / f"{serial}.yaml").read_text(encoding="utf-8")
    )

    return {
        "serial":               serial,
        "fov_width_cm":         cfg.get("fov_width_cm"),
        "crop_roi":             cfg.get("crop_roi"),
        "lines_per_cm_mean":    spectral_lpc,
        "lines_per_cm_median":  phys.get("lines_per_cm_median"),
        "interval_mm_mean":     spectral_interval_cm * 10 if spectral_interval_cm is not None else None,
        "n_peaks":              iv.get("n_peaks"),
        "wire_fwhm_mm_median":  fwhm_d.get("median"),
        "wire_fwhm_mm_ci_lo":   ci[0],
        "wire_fwhm_mm_ci_hi":   ci[1],
        "self_contrast_z":      z,
        "self_contrast_rel_pct":rel_pct,
        "split_half_lpc":       sh_lpc,
        "split_half_period_cm_std": sh_std,
    }


records = []
print(f"{'Serial':<22} {'GT':>6} {'Src':>6} {'Spectral':>9} {'Err%':>7} {'z':>7}  Status")
print("-" * 82)

for serial, spreadsheet_gt, spreadsheet_note in DATASETS:
    try:
        p = load_pipeline(serial)
    except Exception as e:
        print(f"{serial:<22}  ERROR: {e}")
        records.append({"serial": serial, "error": str(e),
                        "spreadsheet_gt_lines_per_cm": spreadsheet_gt,
                        "spreadsheet_gt_note": spreadsheet_note})
        continue

    manual_path = Path("results") / serial / "manual_gt.json"
    manual = json.loads(manual_path.read_text(encoding="utf-8")) if manual_path.exists() else None
    manual_gt = manual.get("lpc_mean") if manual else None
    gt = manual_gt if manual_gt is not None else spreadsheet_gt
    gt_source = "manual" if manual_gt is not None else "sheet"

    lpc  = p["lines_per_cm_mean"]
    err  = (lpc - gt) / gt * 100 if lpc else None
    z    = p["self_contrast_z"]

    flags = []
    if z is not None and abs(z) < 2:  flags.append("LOW-z")
    if err is not None and abs(err) > 25: flags.append("BIG-ERR")

    z_s = f"{z:>7.2f}" if z is not None else "    n/a"
    print(f"{serial:<22} {gt:>6.2f} {gt_source:>6} {lpc:>9.2f}"
          f" {err:>+7.1f}% {z_s}  {' '.join(flags)}")

    records.append({
        "serial":            serial,
        "gt_lines_per_cm":   gt,
        "gt_source":         gt_source,
        "manual_gt_lines_per_cm": manual_gt,
        "spreadsheet_gt_lines_per_cm": spreadsheet_gt,
        "spreadsheet_gt_note": spreadsheet_note,
        "error_pct":         round(err, 2) if err is not None else None,
        **{k: (round(v, 4) if isinstance(v, float) else v)
           for k, v in p.items() if k != "serial"},
    })

out = Path("results") / "pipeline_vs_spreadsheet.json"
out.write_text(json.dumps(records, indent=2, default=str), encoding="utf-8")
print(f"\nSaved -> {out}  ({len(records)} records)")
