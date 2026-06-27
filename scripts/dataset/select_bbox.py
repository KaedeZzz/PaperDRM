"""
Interactive paper crop_roi selector with zoom / pan.

Controls
--------
  Left-click + drag     : draw bounding box
  Mouse wheel           : zoom in / out (centred on cursor)
  Middle-click + drag   : pan
  Right-click + drag    : pan  (alternative)
  R                     : reset current box
  C / Enter             : confirm & save, advance to next image
  S                     : skip (keep existing crop_roi unchanged)
  Q / Escape            : quit
"""
import sys, yaml
sys.path.insert(0, ".")
import cv2, numpy as np
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

BOX_COLOR   = (0,  80, 255)
DRAG_COLOR  = (0, 220, 220)
EXIST_COLOR = (80, 200,  80)
ZOOM_STEP   = 1.25


def _screen_size():
    try:
        import ctypes
        u = ctypes.windll.user32
        return u.GetSystemMetrics(0), u.GetSystemMetrics(1)
    except Exception:
        return 1920, 1080


class BBoxSelector:
    def __init__(self, image_bgr, existing_crop, serial, orig_shape):
        self.img        = image_bgr          # full-res BGR (may be downscaled for memory)
        self.existing   = existing_crop
        self.serial     = serial
        self.orig_h, self.orig_w = orig_shape

        ih, iw = self.img.shape[:2]
        self.img_w, self.img_h = iw, ih

        sw, sh = _screen_size()
        # leave room for title + help bars and taskbar
        avail_w = sw - 20
        avail_h = sh - 120
        fit_scale = min(avail_w / iw, avail_h / ih)

        self.scale  = fit_scale   # current view scale (screen px per img px)
        self.fit_scale = fit_scale
        self.ox     = 0.0         # view offset in image coordinates (top-left of viewport)
        self.oy     = 0.0

        # bbox in image coords (None until drawn)
        self.box_p0 = None   # (ix, iy)
        self.box_p1 = None

        # interaction state
        self._ldown = False
        self._pan_down = False
        self._pan_last = None

        self.confirmed = False
        self.skip      = False
        self.quit      = False

        self.win = f"Crop selector — {serial}"

    # ── coordinate transforms ────────────────────────────────────────────────
    def s2i(self, sx, sy):
        """Screen → image coords."""
        return sx / self.scale + self.ox, sy / self.scale + self.oy

    def i2s(self, ix, iy):
        """Image → screen coords."""
        return (ix - self.ox) * self.scale, (iy - self.oy) * self.scale

    def _clamp_view(self):
        vw = self._vw() / self.scale
        vh = self._vh() / self.scale
        self.ox = max(0.0, min(self.ox, self.img_w - vw))
        self.oy = max(0.0, min(self.oy, self.img_h - vh))

    def _vw(self): return int(self.img_w * self.scale)
    def _vh(self): return int(self.img_h * self.scale)

    # ── mouse callback ───────────────────────────────────────────────────────
    def mouse_cb(self, event, sx, sy, flags, _):
        ix, iy = self.s2i(sx, sy)

        # --- zoom (wheel) ---
        if event == cv2.EVENT_MOUSEWHEEL:
            factor = ZOOM_STEP if flags > 0 else 1.0 / ZOOM_STEP
            new_scale = max(self.fit_scale * 0.5,
                            min(self.scale * factor, self.fit_scale * 20))
            # keep cursor fixed in image space
            self.ox = ix - sx / new_scale
            self.oy = iy - sy / new_scale
            self.scale = new_scale
            self._clamp_view()
            return

        # --- pan (middle or right button) ---
        if event in (cv2.EVENT_MBUTTONDOWN, cv2.EVENT_RBUTTONDOWN):
            self._pan_down = True
            self._pan_last = (sx, sy)
            return
        if event in (cv2.EVENT_MBUTTONUP, cv2.EVENT_RBUTTONUP):
            self._pan_down = False
            self._pan_last = None
            return
        if event == cv2.EVENT_MOUSEMOVE and self._pan_down and self._pan_last:
            dx = (sx - self._pan_last[0]) / self.scale
            dy = (sy - self._pan_last[1]) / self.scale
            self.ox -= dx
            self.oy -= dy
            self._clamp_view()
            self._pan_last = (sx, sy)
            return

        # --- draw bbox (left button) ---
        if event == cv2.EVENT_LBUTTONDOWN:
            self._ldown = True
            self.box_p0 = (ix, iy)
            self.box_p1 = (ix, iy)
        elif event == cv2.EVENT_MOUSEMOVE and self._ldown:
            self.box_p1 = (ix, iy)
        elif event == cv2.EVENT_LBUTTONUP:
            self._ldown = False
            self.box_p1 = (ix, iy)

    # ── render ───────────────────────────────────────────────────────────────
    def _render(self):
        vw = max(1, int(self.img_w * self.scale))
        vh = max(1, int(self.img_h * self.scale))

        # crop the visible portion of the image and scale it
        ox_px = int(self.ox)
        oy_px = int(self.oy)
        # how many image pixels fit in the viewport
        vis_iw = int(vw / self.scale) + 2
        vis_ih = int(vh / self.scale) + 2
        ox_px = max(0, min(ox_px, self.img_w - 1))
        oy_px = max(0, min(oy_px, self.img_h - 1))
        x2 = min(self.img_w, ox_px + vis_iw)
        y2 = min(self.img_h, oy_px + vis_ih)

        patch = self.img[oy_px:y2, ox_px:x2]
        pw = max(1, int(patch.shape[1] * self.scale))
        ph = max(1, int(patch.shape[0] * self.scale))
        view = cv2.resize(patch, (pw, ph), interpolation=cv2.INTER_LINEAR)

        # offset of this patch in screen space
        screen_ox = int((ox_px - self.ox) * self.scale)
        screen_oy = int((oy_px - self.oy) * self.scale)

        sw, sh = _screen_size()
        canvas_w = min(pw + abs(screen_ox), sw - 20)
        canvas_h = min(ph + abs(screen_oy), sh - 120)
        canvas = np.zeros((canvas_h, canvas_w, 3), np.uint8)
        dst_x = max(0, screen_ox)
        dst_y = max(0, screen_oy)
        src_x = max(0, -screen_ox)
        src_y = max(0, -screen_oy)
        copy_w = min(view.shape[1] - src_x, canvas_w - dst_x)
        copy_h = min(view.shape[0] - src_y, canvas_h - dst_y)
        if copy_w > 0 and copy_h > 0:
            canvas[dst_y:dst_y+copy_h, dst_x:dst_x+copy_w] = \
                view[src_y:src_y+copy_h, src_x:src_x+copy_w]

        def _i2c(ix, iy):
            """Image → canvas screen coords."""
            sx = int((ix - self.ox) * self.scale)
            sy = int((iy - self.oy) * self.scale)
            return sx, sy

        # existing box (green dashed)
        if self.existing:
            ex, ey, ew, eh = self.existing
            # scale existing to img coords (existing is in original full-res)
            scale_to_img = self.img_w / self.orig_w
            ex2 = int(ex * scale_to_img); ey2 = int(ey * scale_to_img)
            ew2 = int(ew * scale_to_img); eh2 = int(eh * scale_to_img)
            a, b = _i2c(ex2, ey2)
            c, d = _i2c(ex2 + ew2, ey2 + eh2)
            for i in range(a, c, 24):
                cv2.line(canvas, (i, b), (min(i+12, c), b), EXIST_COLOR, 1)
                cv2.line(canvas, (i, d), (min(i+12, c), d), EXIST_COLOR, 1)
            for i in range(b, d, 24):
                cv2.line(canvas, (a, i), (a, min(i+12, d)), EXIST_COLOR, 1)
                cv2.line(canvas, (c, i), (c, min(i+12, d)), EXIST_COLOR, 1)

        # current box
        if self.box_p0 and self.box_p1:
            ix0, iy0 = self.box_p0
            ix1, iy1 = self.box_p1
            p0s = _i2c(min(ix0,ix1), min(iy0,iy1))
            p1s = _i2c(max(ix0,ix1), max(iy0,iy1))
            color = DRAG_COLOR if self._ldown else BOX_COLOR
            cv2.rectangle(canvas, p0s, p1s, color, 2)
            if not self._ldown:
                fw = int(abs(ix1-ix0) * self.orig_w / self.img_w)
                fh = int(abs(iy1-iy0) * self.orig_h / self.img_h)
                cv2.putText(canvas, f"{fw} x {fh} px",
                            (p0s[0]+4, p0s[1]+18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, BOX_COLOR, 1, cv2.LINE_AA)

        # zoom level indicator
        zoom_pct = int(self.scale / self.fit_scale * 100)
        cv2.putText(canvas, f"zoom {zoom_pct}%", (6, canvas_h - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150,150,150), 1)

        # title + help
        cw = canvas.shape[1]
        tbar = np.zeros((26, cw, 3), np.uint8)
        ex_str = str(self.existing) if self.existing else "none"
        cv2.putText(tbar, f"{self.serial}  |  existing: {ex_str}  |  "
                    f"Wheel=zoom  Mid/RClick-drag=pan  R=reset  C=confirm  S=skip  Q=quit",
                    (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (220,220,80), 1, cv2.LINE_AA)
        return np.vstack([tbar, canvas])

    # ── main loop ────────────────────────────────────────────────────────────
    def run(self):
        cv2.namedWindow(self.win, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.win, self.mouse_cb)

        while True:
            cv2.imshow(self.win, self._render())
            key = cv2.waitKey(30) & 0xFF

            if key in (ord('c'), 13):
                if self.box_p0 and self.box_p1 and \
                        abs(self.box_p1[0]-self.box_p0[0]) > 5 and \
                        abs(self.box_p1[1]-self.box_p0[1]) > 5:
                    self.confirmed = True
                    break
                else:
                    print("  Draw a box first.")
            elif key == ord('s'):
                self.skip = True; break
            elif key in (ord('q'), 27):
                self.quit = True; break
            elif key == ord('r'):
                self.box_p0 = None; self.box_p1 = None

        cv2.destroyWindow(self.win)


def run_selector(serial: str, cfg_path: Path) -> bool:
    try:
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[{serial}] config error: {e}"); return False

    img_path = cfg.get("image_path", "")
    raw = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if raw is None:
        print(f"[{serial}] cannot load {img_path}"); return False

    orig_h, orig_w = raw.shape

    # Downsample large images to at most 4096px wide to save memory
    MEM_MAX = 4096
    mem_scale = min(1.0, MEM_MAX / orig_w)
    if mem_scale < 1.0:
        mw = int(orig_w * mem_scale)
        mh = int(orig_h * mem_scale)
        disp_img = cv2.resize(raw, (mw, mh), interpolation=cv2.INTER_AREA)
    else:
        disp_img = raw
    bgr = cv2.cvtColor(disp_img, cv2.COLOR_GRAY2BGR)

    existing = cfg.get("crop_roi")
    print(f"\n[{serial}]  {orig_w}x{orig_h}  existing={existing}")

    sel = BBoxSelector(bgr, existing, serial, (orig_h, orig_w))
    sel.run()

    if sel.quit:
        return True
    if sel.skip:
        print(f"  Skipped.")
        return False

    # Convert drawn box (in disp_img coords) → original full-res coords
    ix0, iy0 = sel.box_p0
    ix1, iy1 = sel.box_p1
    # disp_img → orig
    to_orig = orig_w / sel.img_w
    fx  = int(round(min(ix0, ix1) * to_orig))
    fy  = int(round(min(iy0, iy1) * to_orig))
    fx2 = int(round(max(ix0, ix1) * to_orig))
    fy2 = int(round(max(iy0, iy1) * to_orig))
    fx  = max(0, min(fx,  orig_w - 1))
    fy  = max(0, min(fy,  orig_h - 1))
    fw  = min(fx2 - fx, orig_w - fx)
    fh  = min(fy2 - fy, orig_h - fy)
    new_crop = [fx, fy, fw, fh]

    # Recompute fov_width_cm: back-calculate paper physical width from old values
    old_fov      = cfg["fov_width_cm"]
    old_crop_w   = existing[2] if existing else orig_w
    paper_width_cm = old_fov * old_crop_w / orig_w
    new_fov = round(orig_w * paper_width_cm / fw, 4)

    print(f"  crop_roi: {new_crop}   fov_width_cm: {new_fov}")

    # Patch config file in-place (preserve comments / ordering)
    text = cfg_path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    out, crop_done, fov_done = [], False, False
    for line in lines:
        if line.startswith("crop_roi:"):
            out.append(f"crop_roi: {new_crop}\n"); crop_done = True
        elif line.startswith("fov_width_cm:"):
            out.append(f"fov_width_cm: {new_fov}\n"); fov_done = True
        else:
            out.append(line)
    # insert if keys were absent
    if not fov_done or not crop_done:
        final = []
        for line in out:
            final.append(line)
            if not fov_done and line.startswith("image_path:"):
                final.append(f"fov_width_cm: {new_fov}\n"); fov_done = True
            if not crop_done and fov_done and "fov_width_cm" in line:
                final.append(f"crop_roi: {new_crop}\n"); crop_done = True
        out = final

    cfg_path.write_text("".join(out), encoding="utf-8")
    return False


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("serials", nargs="*", help="Serials to process (default: all)")
    args = parser.parse_args()
    targets = args.serials if args.serials else DATASETS

    for serial in targets:
        cfg_path = Path("configs") / f"{serial}.yaml"
        if not cfg_path.exists():
            print(f"[{serial}] config not found, skipping"); continue
        if run_selector(serial, cfg_path):
            print("Quit."); break

    print("\nDone. Run  python scripts/make_bbox_overlays.py  to refresh overlays.")
