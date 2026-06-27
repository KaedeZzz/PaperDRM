"""
GT Builder — interactive ground-truth annotation app.

Two phases per folio:
  Phase 1 – Bbox selection  (uses BBoxSelector from select_bbox.py)
  Phase 2 – Laid-line annotation → results/<serial>/manual_gt.json

Annotation controls (Phase 2):
  Left-click          : place horizontal marker on a laid line
  D                   : delete marker nearest to cursor (Y-axis)
  Mouse wheel         : zoom in / out
  Middle/Right drag   : pan
  C / Enter           : confirm & save, next folio
  S                   : skip this folio (keep existing manual_gt if any)
  Q / Escape          : quit

Usage:
  python scripts/gt_builder.py                      # all folios, both phases
  python scripts/gt_builder.py Hh2-12_f190          # single folio
  python scripts/gt_builder.py --skip-bbox          # annotation only
  python scripts/gt_builder.py --skip-annotate      # bbox only
"""
import sys, yaml, json, importlib.util
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, ".")

# ── import BBoxSelector / run_selector from select_bbox ──────────────────────
_spec = importlib.util.spec_from_file_location(
    "select_bbox", Path(__file__).parent / "select_bbox.py"
)
_sb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_sb)
BBoxSelector = _sb.BBoxSelector
run_selector = _sb.run_selector
ZOOM_STEP    = _sb.ZOOM_STEP


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

MARKER_COLOR = (0, 220, 255)   # cyan
CURSOR_COLOR = (80, 200, 80)   # green
STAT_COLOR   = (220, 220, 80)  # yellow


def _screen_size():
    try:
        import ctypes
        u = ctypes.windll.user32
        return u.GetSystemMetrics(0), u.GetSystemMetrics(1)
    except Exception:
        return 1920, 1080


# ─────────────────────────────────────────────────────────────────────────────
#  LineAnnotator
# ─────────────────────────────────────────────────────────────────────────────

class LineAnnotator:
    """
    Zoom/pan viewer for laid-line annotation.
    Laid lines are horizontal in the image, so markers are horizontal lines
    placed at the Y-coordinate of each clicked point.
    """

    def __init__(self, cropped_gray: np.ndarray, cm_per_px: float,
                 serial: str, existing_markers=None):
        self.img = cv2.cvtColor(cropped_gray, cv2.COLOR_GRAY2BGR)
        self.cm_per_px = cm_per_px
        self.serial    = serial

        ih, iw = cropped_gray.shape[:2]
        self.img_w = iw
        self.img_h = ih

        sw, sh = _screen_size()
        fit = min((sw - 20) / iw, (sh - 120) / ih)
        self.scale     = fit
        self.fit_scale = fit
        self.ox = self.oy = 0.0

        # markers: sorted list of Y-coordinates in image space
        self.markers: list[float] = (
            list(existing_markers) if existing_markers else []
        )

        self._pan_down   = False
        self._pan_last   = None
        self._cursor_iy: float | None = None   # current cursor Y in image space

        self.confirmed = False
        self.skip      = False
        self.quit      = False

        self.win = f"Laid-line annotator — {serial}"

    # ── helpers ──────────────────────────────────────────────────────────────

    def _s2i(self, sx, sy):
        return sx / self.scale + self.ox, sy / self.scale + self.oy

    def _iy2sy(self, iy: float) -> int:
        return int((iy - self.oy) * self.scale)

    def _clamp(self):
        sw, sh = _screen_size()
        max_ox = max(0.0, self.img_w - (sw - 20) / self.scale)
        max_oy = max(0.0, self.img_h - (sh - 120) / self.scale)
        self.ox = max(0.0, min(self.ox, max_ox))
        self.oy = max(0.0, min(self.oy, max_oy))

    # ── mouse ────────────────────────────────────────────────────────────────

    def mouse_cb(self, event, sx, sy, flags, _):
        ix, iy = self._s2i(sx, sy)
        self._cursor_iy = iy

        if event == cv2.EVENT_MOUSEWHEEL:
            factor = ZOOM_STEP if flags > 0 else 1.0 / ZOOM_STEP
            new_s  = max(self.fit_scale * 0.5,
                         min(self.scale * factor, self.fit_scale * 30))
            self.ox = ix - sx / new_s
            self.oy = iy - sy / new_s
            self.scale = new_s
            self._clamp()
            return

        if event in (cv2.EVENT_MBUTTONDOWN, cv2.EVENT_RBUTTONDOWN):
            self._pan_down = True
            self._pan_last = (sx, sy)
            return
        if event in (cv2.EVENT_MBUTTONUP, cv2.EVENT_RBUTTONUP):
            self._pan_down = False
            return
        if event == cv2.EVENT_MOUSEMOVE and self._pan_down and self._pan_last:
            dx = (sx - self._pan_last[0]) / self.scale
            dy = (sy - self._pan_last[1]) / self.scale
            self.ox -= dx
            self.oy -= dy
            self._clamp()
            self._pan_last = (sx, sy)
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            self.markers.append(iy)
            self.markers.sort()

    # ── render ───────────────────────────────────────────────────────────────

    def _render(self) -> np.ndarray:
        sw, sh = _screen_size()

        ox_px = max(0, min(int(self.ox), self.img_w - 1))
        oy_px = max(0, min(int(self.oy), self.img_h - 1))
        vis_iw = int((sw - 20) / self.scale) + 2
        vis_ih = int((sh - 120) / self.scale) + 2
        x2 = min(self.img_w, ox_px + vis_iw)
        y2 = min(self.img_h, oy_px + vis_ih)

        patch = self.img[oy_px:y2, ox_px:x2]
        pw = max(1, int(patch.shape[1] * self.scale))
        ph = max(1, int(patch.shape[0] * self.scale))
        view = cv2.resize(patch, (pw, ph), interpolation=cv2.INTER_LINEAR)

        s_ox = int((ox_px - self.ox) * self.scale)
        s_oy = int((oy_px - self.oy) * self.scale)
        cw = min(pw + abs(s_ox), sw - 20)
        ch = min(ph + abs(s_oy), sh - 120)
        canvas = np.zeros((ch, cw, 3), np.uint8)

        dx, dy = max(0, s_ox), max(0, s_oy)
        sx0, sy0 = max(0, -s_ox), max(0, -s_oy)
        cpw = min(view.shape[1] - sx0, cw - dx)
        cph = min(view.shape[0] - sy0, ch - dy)
        if cpw > 0 and cph > 0:
            canvas[dy:dy + cph, dx:dx + cpw] = (
                view[sy0:sy0 + cph, sx0:sx0 + cpw]
            )

        # horizontal markers at each laid line Y position
        for my in self.markers:
            sy_m = self._iy2sy(my)
            if 0 <= sy_m < ch:
                cv2.line(canvas, (0, sy_m), (cw, sy_m), MARKER_COLOR, 1)

        # cursor preview (horizontal)
        if self._cursor_iy is not None:
            sy_c = self._iy2sy(self._cursor_iy)
            if 0 <= sy_c < ch:
                cv2.line(canvas, (0, sy_c), (cw, sy_c), CURSOR_COLOR, 1)

        # stats
        n = len(self.markers)
        zoom_pct = int(self.scale / self.fit_scale * 100)
        if n >= 2:
            gaps = np.diff(sorted(self.markers))
            mg  = float(np.mean(gaps))
            lpc = 1.0 / (mg * self.cm_per_px) if mg > 0 else 0.0
            stat = (f"N={n}  mean_gap={mg:.1f}px  lpc={lpc:.2f}/cm  "
                    f"zoom={zoom_pct}%  D=del-nearest")
        else:
            stat = f"N={n}  (需要标记至少2根线)  zoom={zoom_pct}%  D=del"
        cv2.putText(canvas, stat, (6, ch - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, STAT_COLOR, 1, cv2.LINE_AA)

        # title bar
        tbar = np.zeros((28, cw, 3), np.uint8)
        title = (f"{self.serial}  cm/px={self.cm_per_px:.5f}  |  "
                 f"左键=标线  D=删除  滚轮=缩放  中键/右键拖=平移  "
                 f"C=确认  S=跳过  Q=退出")
        cv2.putText(tbar, title, (6, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 80), 1,
                    cv2.LINE_AA)
        return np.vstack([tbar, canvas])

    # ── main loop ────────────────────────────────────────────────────────────

    def run(self):
        cv2.namedWindow(self.win, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.win, self.mouse_cb)
        while True:
            cv2.imshow(self.win, self._render())
            key = cv2.waitKey(30) & 0xFF

            if key in (ord('c'), 13):
                if len(self.markers) >= 2:
                    self.confirmed = True
                    break
                print("  至少标记2根laid line。")
            elif key == ord('s'):
                self.skip = True
                break
            elif key in (ord('q'), 27):
                self.quit = True
                break
            elif key == ord('d') and self.markers and self._cursor_iy is not None:
                dists = [abs(m - self._cursor_iy) for m in self.markers]
                del self.markers[int(np.argmin(dists))]

        cv2.destroyWindow(self.win)


# ─────────────────────────────────────────────────────────────────────────────
#  Annotation runner
# ─────────────────────────────────────────────────────────────────────────────

def run_annotation(serial: str, cfg_path: Path, out_dir: Path) -> bool:
    """Run Phase 2 for one folio. Returns True if user quit globally."""
    try:
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[{serial}] config error: {e}")
        return False

    raw = cv2.imread(cfg.get("image_path", ""), cv2.IMREAD_GRAYSCALE)
    if raw is None:
        print(f"[{serial}] cannot load image")
        return False

    crop = cfg.get("crop_roi")
    if not crop:
        print(f"[{serial}] no crop_roi — run Phase 1 first")
        return False

    orig_h, orig_w = raw.shape
    fov = cfg.get("fov_width_cm")
    if not fov:
        print(f"[{serial}] no fov_width_cm")
        return False

    cm_per_px = float(fov) / float(orig_w)

    x, y, w, h = crop
    cropped = raw[y:y + h, x:x + w]

    # load existing GT markers if any
    gt_path = out_dir / "manual_gt.json"
    existing_markers = None
    if gt_path.exists():
        try:
            old = json.loads(gt_path.read_text(encoding="utf-8"))
            existing_markers = old.get("y_positions_px", [])
            print(f"  已有GT: {len(existing_markers)} 根线, "
                  f"lpc={old.get('lpc_mean'):.3f}/cm")
        except Exception:
            existing_markers = None

    print(f"  {orig_w}×{orig_h}px  crop [{x},{y},{w},{h}]  cm/px={cm_per_px:.5f}")

    ann = LineAnnotator(cropped, cm_per_px, serial, existing_markers)
    ann.run()

    if ann.quit:
        return True
    if ann.skip:
        print("  已跳过。")
        return False

    srt  = sorted(ann.markers)
    gaps = np.diff(srt).astype(float)
    mg   = float(np.mean(gaps))
    med  = float(np.median(gaps))
    lpc_mean   = 1.0 / (mg  * cm_per_px)
    lpc_median = 1.0 / (med * cm_per_px)

    gt_data = {
        "serial":          serial,
        "n_lines_marked":  len(srt),
        "y_positions_px":  [round(v, 2) for v in srt],
        "mean_gap_px":     round(mg,  3),
        "median_gap_px":   round(med, 3),
        "cm_per_px":       round(cm_per_px, 7),
        "lpc_mean":        round(lpc_mean,   4),
        "lpc_median":      round(lpc_median, 4),
        "crop_wh":         [ann.img_w, ann.img_h],
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    gt_path.write_text(json.dumps(gt_data, indent=2), encoding="utf-8")
    print(f"  GT saved: n={len(srt)}  "
          f"lpc_mean={lpc_mean:.3f}  lpc_median={lpc_median:.3f}")
    print(f"  → {gt_path}")
    return False


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Interactive GT builder: bbox selection + laid-line annotation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("serials", nargs="*",
                    help="Folios to process (default: all)")
    ap.add_argument("--skip-bbox", action="store_true",
                    help="Skip Phase 1 (bbox selection)")
    ap.add_argument("--skip-annotate", action="store_true",
                    help="Skip Phase 2 (line annotation)")
    args = ap.parse_args()

    targets = args.serials if args.serials else DATASETS

    for serial in targets:
        cfg_path = Path("configs") / f"{serial}.yaml"
        out_dir  = Path("results") / serial

        if not cfg_path.exists():
            print(f"[{serial}] config not found, skipping")
            continue

        print(f"\n{'='*60}")
        print(f"  Serial: {serial}")

        # Phase 1 — bbox selection
        if not args.skip_bbox:
            print(f"  Phase 1 — bbox selection")
            if run_selector(serial, cfg_path):
                print("Quit."); break

        # Phase 2 — laid-line annotation
        if not args.skip_annotate:
            print(f"  Phase 2 — laid-line annotation")
            if run_annotation(serial, cfg_path, out_dir):
                print("Quit."); break

    print("\nDone. Compare with:  python scripts/compare_vs_manual_gt.py")
