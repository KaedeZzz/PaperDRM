"""
Collect full-image pipeline measurements + spreadsheet GT for all 9 folios
and save to results/pipeline_vs_spreadsheet.json.
"""
import sys, json, yaml
sys.path.insert(0, ".")
from pathlib import Path

DATASETS = [
    # (serial,         gt_lpc,   gt_note)
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
        "lines_per_cm_mean":    phys.get("lines_per_cm_mean"),
        "lines_per_cm_median":  phys.get("lines_per_cm_median"),
        "interval_mm_mean":     phys.get("mean_interval_cm", 0) * 10,
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
print(f"{'Serial':<22} {'GT':>5}  {'Mean':>7}  {'Med':>7}  {'Err%':>7}  {'z':>6}  {'SH_lpc':>7}  Status")
print("-" * 90)

for serial, gt, gt_note in DATASETS:
    try:
        p = load_pipeline(serial)
    except Exception as e:
        print(f"{serial:<22}  ERROR: {e}")
        records.append({"serial": serial, "error": str(e),
                        "gt_lines_per_cm": gt, "gt_note": gt_note})
        continue

    lpc  = p["lines_per_cm_mean"]
    err  = (lpc - gt) / gt * 100 if lpc else None
    z    = p["self_contrast_z"]
    sh   = p["split_half_lpc"]

    flags = []
    if z is not None and abs(z) < 2:  flags.append("LOW-z")
    if err is not None and abs(err) > 25: flags.append("BIG-ERR")

    z_s  = f"{z:>6.2f}"  if z  is not None else "   n/a"
    sh_s = f"{sh:>7.2f}" if sh is not None else "    n/a"
    print(f"{serial:<22} {gt:>5.1f}  {lpc:>7.2f}  {p['lines_per_cm_median']:>7.2f}"
          f"  {err:>+7.1f}%  {z_s}  {sh_s}  {' '.join(flags)}")

    records.append({
        "serial":            serial,
        "gt_lines_per_cm":   gt,
        "gt_note":           gt_note,
        "error_pct":         round(err, 2) if err is not None else None,
        **{k: (round(v, 4) if isinstance(v, float) else v)
           for k, v in p.items() if k != "serial"},
    })

out = Path("results") / "pipeline_vs_spreadsheet.json"
out.write_text(json.dumps(records, indent=2, default=str), encoding="utf-8")
print(f"\nSaved -> {out}  ({len(records)} records)")
