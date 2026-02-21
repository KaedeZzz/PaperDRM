import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from .settings import CacheConfig, DRPConfig, load_cache_config, load_drp_config, save_cache_config
from .paths import DataPaths


def resolve_config_path(config_path: str | Path | None) -> Path:
    """
    Resolve the DRP config path relative to the repository root when not absolute.
    """
    repo_root = Path(__file__).resolve().parent.parent
    candidate = Path(config_path) if config_path is not None else repo_root / "exp_param.yaml"
    return candidate if candidate.is_absolute() else repo_root / candidate


def resolve_image_folder(folder: str | Path | None, paths: DataPaths) -> Path:
    """
    Resolve the folder where images are located, defaulting to the processed path.
    """
    if folder is None:
        return paths.processed
    folder_path = Path(folder)
    return folder_path if folder_path.is_absolute() else paths.root / folder_path


def load_images(folder: Path, img_format: str, num_workers: int | None = None) -> list[np.ndarray]:
    """
    Load grayscale images from a folder matching the given extension.
    """
    image_paths = sorted(folder.glob(f"*.{img_format}"))
    if not image_paths:
        raise ValueError(f"No images with extension .{img_format} found in {folder}")

    def _read(path: Path) -> np.ndarray:
        img = cv2.imread(str(path), flags=cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise IOError(f"Could not open image at path: {path}")
        return img

    # Threaded decode helps when disk and CPU can overlap; falls back to serial otherwise.
    workers = num_workers if num_workers is not None else (os.cpu_count() or 1)
    if workers <= 1:
        return [_read(p) for p in tqdm(image_paths, desc="loading images")]

    with ThreadPoolExecutor(max_workers=workers) as pool:
        images = list(tqdm(pool.map(_read, image_paths), total=len(image_paths), desc="loading images"))
    return images


def open_drp_memmap(path: Path, shape: tuple[int, int, int, int], mode: str) -> np.memmap:
    """
    Open a memmap for the DRP stack, creating parent directories if needed.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    # On Windows, opening an already-mapped file with mode='w+' (truncate)
    # can fail when another process still holds a map. To avoid destructive
    # truncation, reuse the existing file in r+ and only expand if needed.
    if mode == "w+" and path.exists():
        required_bytes = int(np.prod(shape, dtype=np.int64)) * np.dtype("uint8").itemsize
        current_bytes = path.stat().st_size
        if current_bytes < required_bytes:
            with path.open("r+b") as fh:
                fh.truncate(required_bytes)
        return np.memmap(path, dtype="uint8", mode="r+", shape=shape)  # type: ignore

    return np.memmap(path, dtype="uint8", mode=mode, shape=shape)  # type: ignore


def prepare_cache(
    paths: DataPaths,
    angle_slice: tuple[int, int],
    stack_shape: tuple[int, int, int, int],
    data_serial: str | int | None,
) -> tuple[np.memmap, CacheConfig, bool]:
    """
    Prepare the DRP cache. Returns (memmap, cache_cfg, is_new_stack).
    is_new_stack is True when the memmap is opened in write mode and needs population.
    """
    data_config_path = paths.cache / "data_config.yaml"
    drp_path = paths.cache / "drp.dat"
    cache_cfg = CacheConfig(
            ph_slice=angle_slice[0], 
            th_slice=angle_slice[1], 
            data_serial=data_serial
    )
    if data_config_path.exists() and drp_path.exists():
        # DRP data exists; check if data serial matches
        existing_cfg = load_cache_config(data_config_path)
        if existing_cfg.data_serial == data_serial:
            # Use existing data
            cache_cfg = existing_cfg
            memmap = open_drp_memmap(drp_path, stack_shape, mode="r+")
            return memmap, cache_cfg, False
        # Data serial changed: overwrite in-place without deleting (avoids Windows file-lock issues)
        save_cache_config(data_config_path, cache_cfg)
        memmap = open_drp_memmap(drp_path, stack_shape, mode="w+")
        return memmap, cache_cfg, True

    # Create new cache
    save_cache_config(data_config_path, cache_cfg)
    memmap = open_drp_memmap(drp_path, stack_shape, mode="w+")
    return memmap, cache_cfg, True
