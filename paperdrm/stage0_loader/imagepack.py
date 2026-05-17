from pathlib import Path
import warnings

import cv2
import numpy as np

from dataclasses import replace

from paperdrm.stage0_loader.settings import (
    CacheConfig,
    DRPConfig,
    Settings,
    load_drp_config,
    resolve_drp_from_yaml,
    save_cache_config,
)
from paperdrm.stage0_loader.image_io import (
    open_drp_memmap,
    prepare_cache,
    resolve_config_path,
    resolve_image_folder,
    load_images,
    load_images_from_paths,
)
from paperdrm.stage0_loader.inference import infer_drp_config_from_folder, verify_drp_match
from paperdrm.stage0_loader.paths import DataPaths
from paperdrm.stage0_drp import (
    apply_angle_slice,
    apply_theta_min_filter,
    build_drp_stack,
    drp_from_images,
    drp_from_stack,
    mask_images as compute_mask_images,
    mean_drp_from_stack,
)
from paperdrm.stage4_viz.drp import plot_drp


class ImagePack:
    def __init__(
        self,
        folder: str | Path | None = None,
        img_format: str = "jpg",
        angle_slice: tuple[int, int] = (1, 1),
        data_root: str | Path = "data",
        config_path: str | Path | None = None,
        use_cached_stack: bool = True,
        load_workers: int | None = None,
        subtract_background: bool = True,
        subtraction_scale_percentile: float = 99.5,
        square_crop: bool = False,
        verbose: bool | None = None,
        settings: Settings | None = None,
    ):
        # Prefer a single Settings object; reject mixed kwargs when provided.
        if settings is not None:
            overrides = any(
                [
                    folder is not None,
                    img_format != "jpg",
                    angle_slice != (1, 1),
                    data_root != "data",
                    config_path is not None,
                    use_cached_stack is not True,
                    load_workers is not None,
                    subtract_background is not True,
                    subtraction_scale_percentile != 99.5,
                    square_crop is not False,
                    verbose is not None,
                ]
            )
            if overrides:
                raise ValueError("When providing settings, do not pass additional loader kwargs.")
            self.settings = settings
        else:
            cfg_path = resolve_config_path(config_path)
            drp_cfg, data_serial_hint = resolve_drp_from_yaml(cfg_path)
            self.settings = Settings(
                data_root=data_root,
                folder=folder,
                img_format=img_format,
                angle_slice=angle_slice,
                use_cached_stack=use_cached_stack,
                subtract_background=subtract_background,
                subtraction_scale_percentile=subtraction_scale_percentile,
                load_workers=load_workers,
                config_path=cfg_path,
                drp=drp_cfg,
                data_serial_hint=data_serial_hint,
                square_crop=square_crop,
                verbose=verbose if verbose is not None else False,
            )

        self.verbose = self.settings.verbose
        self.paths = DataPaths.from_root(self.settings.data_root)
        self.folder = resolve_image_folder(self.settings.folder, self.paths)
        self.config_path = self.settings.config_path or resolve_config_path(None)

        # Either populate DRPConfig by inference (yaml omitted the six acq
        # fields) or cross-check yaml-provided values against the actual files.
        if self.settings.drp is None:
            inferred, report = infer_drp_config_from_folder(
                self.folder, img_format=self.settings.img_format, strict=True
            )
            serial = self.settings.data_serial_hint
            if serial is None:
                serial = _data_serial_from_folder(self.folder)
                if serial is not None:
                    self._log(f"data_serial inferred from folder name: {serial!r}")
            inferred = replace(inferred, data_serial=serial)
            self.settings = self.settings.with_overrides(drp=inferred, data_serial_hint=None)
            self._log(f"Inferred DRPConfig from {self.folder}:\n{report.summary()}")
        else:
            inferred, _ = infer_drp_config_from_folder(
                self.folder, img_format=self.settings.img_format, strict=True
            )
            verify_drp_match(self.settings.drp, inferred, source=self.folder)

        self.base_config: DRPConfig = self.settings.drp  # type: ignore[assignment]
        self.data_serial = self.base_config.data_serial
        self._log(f"Initialising ImagePack with data_root={self.paths.root} folder={self.folder}")

        # Pre-select which image paths to load: apply angle_slice + theta_min before
        # reading pixels so we never load the full angular grid into memory.
        all_image_paths = sorted(self.folder.glob(f"*.{self.settings.img_format}"))
        load_paths, sliced_cfg = apply_angle_slice(all_image_paths, self.base_config, self.settings.angle_slice)
        load_paths, filtered_cfg = apply_theta_min_filter(load_paths, sliced_cfg, self.settings.theta_min_deg)
        self._log(
            f"Angular pre-filter: {len(all_image_paths)} -> {len(load_paths)} images "
            f"(angle_slice={self.settings.angle_slice}, theta_min={self.settings.theta_min_deg})"
        )

        # Load only the selected images
        self._log(f"Loading {len(load_paths)} images ({self.settings.img_format}) from {self.folder}")
        self.images = load_images_from_paths(load_paths, num_workers=self.settings.load_workers)
        self._log(f"Loaded {len(self.images)} images; first image shape {self.images[0].shape}")

        # Optional brightness-invariant preprocessing: subtract blurred backgrounds.
        # If a pre-computed background folder exists, load from disk (streaming).
        # Otherwise compute Gaussian blur on-the-fly — no manual bg_blur.py run needed.
        if self.settings.subtract_background:
            sibling_bg = self.folder / "background"
            global_bg = self.paths.root / "background"
            if sibling_bg.exists():
                bg_folder: Path | None = sibling_bg
            elif global_bg.exists():
                bg_folder = global_bg
            else:
                bg_folder = None

            if bg_folder is not None:
                self._log(f"Subtracting backgrounds from {bg_folder} (streaming)")
            else:
                self._log("No background folder found — computing Gaussian blur on-the-fly (sigma=100)")

            subtracted: list[np.ndarray] = []
            for img, img_path in zip(self.images, load_paths):
                if bg_folder is not None:
                    bg = cv2.imread(str(bg_folder / img_path.name), cv2.IMREAD_GRAYSCALE)
                    if bg is None:
                        raise IOError(f"Could not open background image: {bg_folder / img_path.name}")
                    if img.shape != bg.shape:
                        raise ValueError(f"Background shape {bg.shape} does not match image shape {img.shape}.")
                else:
                    bg = cv2.GaussianBlur(img, (0, 0), sigmaX=100, borderType=cv2.BORDER_REFLECT_101)
                diff = img.astype(np.float32) - bg.astype(np.float32)
                diff = np.clip(diff, 0, None)
                ref = np.percentile(diff, self.settings.subtraction_scale_percentile)
                scale = 255.0 / max(ref, 1.0)
                diff = np.clip(diff * scale, 0, 255).astype(np.uint8)
                subtracted.append(diff)
            self.images = subtracted

        if self.settings.square_crop:
            self.images = self._crop_to_square(self.images)
            self._log(f"Applied square crop -> new shape {self.images[0].shape}")

        self.param = filtered_cfg
        self.angle_slice = self.settings.angle_slice
        self.num_images = len(self.images)
        self.h, self.w = self.images[0].shape
        self._log(
            "Applied angular filtering -> "
            f"ph_num={self.param.ph_num}, th_num={self.param.th_num}, "
            f"th_min={self.param.th_min}, th_max={self.param.th_max}"
        )

        stack_shape = (self.h, self.w, self.param.ph_num, self.param.th_num)
        self._log(f"Preparing cache for angle_slice={self.angle_slice} stack_shape={stack_shape}")
        self.drp_stack, cache_cfg, stack_needs_build = prepare_cache(
            self.paths, self.angle_slice, stack_shape, self.data_serial
        )

        cache_slice = (cache_cfg.ph_slice, cache_cfg.th_slice)
        if cache_slice != self.angle_slice:
            self._log(f"Cache slice {cache_slice} != requested {self.angle_slice}; recreating memmap")
            self._close_memmap(self.drp_stack)
            self.drp_stack = open_drp_memmap(
                self.paths.cache / "drp.dat",
                mode="w+",
                shape=stack_shape,
            )
            save_cache_config(
                self.paths.cache / "data_config.yaml",
                CacheConfig(
                    ph_slice=self.angle_slice[0],
                    th_slice=self.angle_slice[1],
                    data_serial=self.data_serial,
                ),
            )
            stack_needs_build = True

        if self.drp_stack.shape != stack_shape or not self.settings.use_cached_stack:
            reason = "shape mismatch" if self.drp_stack.shape != stack_shape else "use_cached_stack=False"
            self._log(f"Recreating DRP memmap due to {reason}; expected {stack_shape}, found {self.drp_stack.shape}")
            self._close_memmap(self.drp_stack)
            self.drp_stack = open_drp_memmap(
                self.paths.cache / "drp.dat",
                mode="w+",
                shape=stack_shape,
            )
            save_cache_config(
                self.paths.cache / "data_config.yaml",
                CacheConfig(
                    ph_slice=self.angle_slice[0],
                    th_slice=self.angle_slice[1],
                    data_serial=self.data_serial,
                ),
            )
            stack_needs_build = True

        if stack_needs_build:
            self._log("Building DRP stack into cache")
            build_drp_stack(self.images, self.param, self.drp_stack, verbose=self.verbose)
        self._log("ImagePack initialisation complete")

    def __iter__(self):
        return iter((self.images, self.param))

    def slice_images(self, angle_slice: tuple[int, int]):
        self.images, self.param = apply_angle_slice(self.images, self.base_config, angle_slice)
        self.num_images = len(self.images)
        return self.images

    def mask_images(self, mask: np.ndarray, normalize: bool = False):
        self.images = compute_mask_images(self.images, mask, normalize)
        return self.images

    def _log(self, message: str) -> None:
        if self.verbose:
            print(f"[ImagePack] {message}")

    def _crop_to_square(self, images: list[np.ndarray]) -> list[np.ndarray]:
        h, w = images[0].shape
        if h == w:
            return images
        size = min(h, w)
        start_h = (h - size) // 2
        start_w = (w - size) // 2
        cropped: list[np.ndarray] = []
        for img in images:
            cropped.append(img[start_h : start_h + size, start_w : start_w + size])
        return cropped

    def drp(self, loc, mode: str = "kernel"):
        if mode == "pixel":
            return drp_from_images(self.images, self.param, loc)
        if mode == "kernel":
            return drp_from_stack(self.drp_stack, loc)
        raise ValueError("mode must be 'pixel' or 'kernel'")

    def plot_drp(self, drp_array, cmap: str = "jet", project: str = "stereo", ax=None):
        return plot_drp(drp_array, self.param, cmap=cmap, project=project, ax=ax)

    def get_mean_drp(self, mode: str = "kernel"):
        if mode == "pixel":
            # Vectorized mean across all pixels: reshape stack to [phi, theta, h, w]
            arr = np.stack(self.images, axis=0).reshape(
                self.param.ph_num, self.param.th_num, self.h, self.w
            )
            return arr.mean(axis=(2, 3))
        return mean_drp_from_stack(self.drp_stack)

    def get_drp_stack(self):
        return self.drp_stack

    @staticmethod
    def _close_memmap(memmap_obj):
        """
        Close a NumPy memmap's underlying mmap, ignoring errors.
        """
        try:
            if hasattr(memmap_obj, "_mmap") and memmap_obj._mmap is not None:
                memmap_obj._mmap.close()
        except Exception:
            pass


def _data_serial_from_folder(folder: Path) -> str | int | None:
    """
    Fall back to the image folder's basename when neither yaml nor hint
    supplied ``data_serial``. Returns int when the basename is numeric,
    otherwise the raw string. Returns None for "obviously generic" names
    (e.g. ``raw``, ``processed``, ``data``) so legacy flat layouts don't
    silently get a meaningless serial like "raw".
    """
    generic = {"raw", "processed", "background", "cache", "data", "datasets"}
    name = folder.name
    if not name or name in generic:
        return None
    try:
        return int(name)
    except ValueError:
        return name
