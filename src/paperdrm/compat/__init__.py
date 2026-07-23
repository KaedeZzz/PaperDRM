"""Compatibility adapters used while V1 paths remain active."""

from paperdrm.compat.v1 import (
    V1ResultDirectoryBackend,
    config_from_settings,
    load_v1_config,
    result_from_directory,
)

__all__ = [
    "V1ResultDirectoryBackend",
    "config_from_settings",
    "load_v1_config",
    "result_from_directory",
]
