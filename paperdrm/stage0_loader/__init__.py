"""
Data loading and DRP stack management.

Submodules:
- settings:  Settings, DRPConfig, CacheConfig dataclasses and YAML I/O
- paths:     DataPaths resolver for data/{raw,processed,background,cache}
- image_io:  load_images, prepare_cache, open_drp_memmap
- imagepack: ImagePack -- top-level entry that ties the above together
"""

from paperdrm.stage0_loader.settings import (
    Settings,
    DRPConfig,
    CacheConfig,
    load_drp_config,
    load_cache_config,
    save_drp_config,
    save_cache_config,
)
from paperdrm.stage0_loader.paths import DataPaths
from paperdrm.stage0_loader.image_io import (
    load_images,
    open_drp_memmap,
    prepare_cache,
    resolve_config_path,
    resolve_image_folder,
)
from paperdrm.stage0_loader.imagepack import ImagePack

__all__ = [
    "Settings",
    "DRPConfig",
    "CacheConfig",
    "DataPaths",
    "ImagePack",
    "load_drp_config",
    "load_cache_config",
    "save_drp_config",
    "save_cache_config",
    "load_images",
    "open_drp_memmap",
    "prepare_cache",
    "resolve_config_path",
    "resolve_image_folder",
]
