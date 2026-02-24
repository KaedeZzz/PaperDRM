from pathlib import Path
import warnings

import numpy as np

from .settings import CacheConfig, DRPConfig, load_drp_config, save_cache_config
from .drp_compute import (
    apply_angle_slice,
    apply_theta_min_filter,
    build_drp_stack,
    drp_from_images,
    drp_from_stack,
    mask_images as compute_mask_images,
    mean_drp_from_stack,
)
from .drp_plot import plot_drp
from .image_io import (
    open_drp_memmap,
    prepare_cache,
    resolve_config_path,
    resolve_image_folder,
    load_images,
)
from .paths import DataPaths
from .settings import Settings


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
            drp_cfg = load_drp_config(cfg_path)
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
                square_crop=square_crop,
                verbose=verbose if verbose is not None else False,
            )

        self.verbose = self.settings.verbose
        self.paths = DataPaths.from_root(self.settings.data_root)
        self.folder = resolve_image_folder(self.settings.folder, self.paths)
        self.config_path = self.settings.config_path or resolve_config_path(None)
        self.base_config: DRPConfig = self.settings.drp  # type: ignore[assignment]
        self.data_serial = self.base_config.data_serial
        self._log(f"Initialising ImagePack with data_root={self.paths.root} folder={self.folder}")

        # Load raw grayscale images from disk
        self._log(f"Loading images ({self.settings.img_format}) from {self.folder}")
        self.images = load_images(self.folder, self.settings.img_format, num_workers=self.settings.load_workers)
        self._log(f"Loaded {len(self.images)} images; first image shape {self.images[0].shape}")

        # Optional brightness-invariant preprocessing: subtract blurred backgrounds
        if self.settings.subtract_background:
            bg_folder = self.paths.root / "background"
            if not bg_folder.exists():
                warnings.warn(f"Background folder not found at {bg_folder}; skipping subtraction.")
            else:
                self._log(f"Subtracting backgrounds from {bg_folder}")
                bg_images = load_images(bg_folder, self.settings.img_format, num_workers=self.settings.load_workers)
                if len(bg_images) != len(self.images):
                    raise ValueError(f"Background count {len(bg_images)} does not match image count {len(self.images)}.")
                subtracted: list[np.ndarray] = []
                for img, bg in zip(self.images, bg_images):
                    if img.shape != bg.shape:
                        raise ValueError(f"Background shape {bg.shape} does not match image shape {img.shape}.")
                    # Subtract then rescale so images share roughly consistent brightness.
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
        self.num_images = len(self.images)
        self.h, self.w = self.images[0].shape

        # Precompute the stack shape for the requested slice to size memmap correctly
        if self.base_config.ph_num % self.settings.angle_slice[0] != 0 or self.base_config.th_num % self.settings.angle_slice[1] != 0:
            raise ValueError("Angle slices must evenly divide ph_num and th_num.")

        sliced_ph = self.base_config.ph_num // self.settings.angle_slice[0]
        sliced_th = self.base_config.th_num // self.settings.angle_slice[1]
        stack_shape = (self.h, self.w, sliced_ph, sliced_th)
        self._log(f"Preparing cache for angle_slice={self.settings.angle_slice} stack_shape={stack_shape}")
        self.drp_stack, cache_cfg, stack_needs_build = prepare_cache(
            self.paths, self.settings.angle_slice, stack_shape, self.data_serial
        )

        # Use caller's slice preference; rebuild cache if it differs from what's stored.
        cache_slice = (cache_cfg.ph_slice, cache_cfg.th_slice)
        self.angle_slice = self.settings.angle_slice
        if cache_slice != self.angle_slice:
            # Force recreation with new slice parameters
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

        # Apply slicing to images and config
        self.images, self.param = apply_angle_slice(self.images, self.base_config, self.angle_slice)
        self.images, self.param = apply_theta_min_filter(self.images, self.param, self.settings.theta_min_deg)
        self.num_images = len(self.images)
        self._log(
            "Applied angular filtering -> "
            f"ph_num={self.param.ph_num}, th_num={self.param.th_num}, "
            f"th_min={self.param.th_min}, th_max={self.param.th_max}"
        )

        expected_shape = (self.h, self.w, self.param.ph_num, self.param.th_num)
        if self.drp_stack.shape != expected_shape or not self.settings.use_cached_stack:
            # Shape mismatch or caller requested rebuild: recreate stack
            reason = "shape mismatch" if self.drp_stack.shape != expected_shape else "use_cached_stack=False"
            self._log(f"Recreating DRP memmap due to {reason}; expected {expected_shape}, found {self.drp_stack.shape}")
            self._close_memmap(self.drp_stack)
            self.drp_stack = open_drp_memmap(
                self.paths.cache / "drp.dat",
                mode="w+",
                shape=expected_shape,
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
