"""
Compare pipeline lpc estimates against manually-annotated ground truth.
Also shows spreadsheet GT for reference.

Usage:
  python scripts/compare_vs_manual_gt.py
  python scripts/compare_vs_manual_gt.py Hh2-12_f190 Kk1-5_f5v
"""
import sys, json
sys.path.insert(0, ".")
from pathlib import Path

DATASETS = [
    "Kk1-5_f5v",
    "Kk1-5_f9v",
    "Hh2-12_f190",
    "Ee5-22_f328r",
    "Ff2-6_f140r",
    "Ff4-9_f42r",
    "Ff4-15_f24r",
    "Hh2-10_f24r",
    "Ii3-8_f135v",
]

SS_GT = {
    "Kk1-5_f5v":    9.0,
    "Kk1-5_f9v":    9.0,
    "Hh2-12_f190":  10.0,
    "Ee5-22_f328r": 10.0,
    "Ff2-6_f140r":  11.0,
    "Ff4-9_f42r":    6.0,
    "Ff4-15_f24r":  13.5,
    "Hh2-10_f24r":  13.5,
    "Ii3-8_f135v":   9.0,
}


def _load(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return None


def main(targets):
    hdr = (f"{'Serial':<22} {'Manual':>8} {'Pipeline':>9} "
           f"{'Err_M%':>8} {'SS_GT':>7} {'Err_SS%':>9}  Flags")
    print(hdr)
    print("-" * len(hdr))

    for serial in targets:
        base = Path("results") / serial
        gt   = _load(base / "manual_gt.json")
        iv   = _load(base / "interval_distribution.json")
        ss   = SS_GT.get(serial)

        gt_lpc = gt["lpc_mean"]                         if gt else None
        pl_lpc = (
            iv["physical"].get(
                "spectral_lines_per_cm",
                iv["physical"].get("lines_per_cm_mean"),
            )
            if iv else None
        )

        err_m  = (pl_lpc - gt_lpc) / gt_lpc * 100 if (gt_lpc and pl_lpc) else None
        err_ss = (pl_lpc - ss)     / ss     * 100 if (ss     and pl_lpc) else None

        gt_s  = f"{gt_lpc:>8.2f}" if gt_lpc is not None else "       —"
        pl_s  = f"{pl_lpc:>9.2f}" if pl_lpc is not None else "        —"
        em_s  = f"{err_m:>+8.1f}%" if err_m  is not None else "        —"
        ss_s  = f"{ss:>7.1f}"      if ss     is not None else "      —"
        es_s  = f"{err_ss:>+9.1f}%" if err_ss is not None else "         —"

        flags = []
        if gt_lpc is None:
            flags.append("NO-MANUAL-GT")
        elif err_m is not None and abs(err_m) > 10:
            flags.append("BIG-ERR")

        print(f"{serial:<22} {gt_s} {pl_s} {em_s} {ss_s} {es_s}  "
              f"{' '.join(flags)}")

    # detail block for annotated folios
    annotated = [s for s in targets
                 if _load(Path("results") / s / "manual_gt.json")]
    if annotated:
        print(f"\n  Manual GT detail")
        print(f"  {'Serial':<22} {'N_marked':>8} {'mean_gap_px':>12} "
              f"{'lpc_mean':>10} {'lpc_med':>9}")
        for serial in annotated:
            gt = _load(Path("results") / serial / "manual_gt.json")
            print(f"  {serial:<22} {gt['n_lines_marked']:>8} "
                  f"{gt['mean_gap_px']:>12.2f} "
                  f"{gt['lpc_mean']:>10.3f} {gt['lpc_median']:>9.3f}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("serials", nargs="*")
    args = ap.parse_args()
    targets = args.serials if args.serials else DATASETS
    main(targets)
