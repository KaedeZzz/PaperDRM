from __future__ import annotations

from dataclasses import dataclass, replace, asdict
from pathlib import Path
from typing import Any

import yaml


# Unified DRP acquisition/cache config (moved from config.py for central access).
@dataclass
class DRPConfig:
    """
    Configuration for DRP acquisition parameters.
    """

    th_min: int
    th_max: int
    th_num: int
    ph_min: int
    ph_max: int
    ph_num: int
    phi_slice: int = 1
    theta_slice: int = 1
    data_serial: str | int | None = None

    def validate(self) -> None:
        if self.ph_min >= self.ph_max:
            raise ValueError("ph_min must be less than ph_max.")
        if self.th_min >= self.th_max:
            raise ValueError("th_min must be less than th_max.")
        if self.ph_num < 2 or self.th_num < 2:
            raise ValueError("ph_num and th_num must be at least 2.")
        if self.phi_slice < 1 or self.theta_slice < 1:
            raise ValueError("phi_slice and theta_slice must be positive.")

    @property
    def ph_step(self) -> float:
        return (self.ph_max - self.ph_min) / (self.ph_num - 1)

    @property
    def th_step(self) -> float:
        return (self.th_max - self.th_min) / (self.th_num - 1)

    def recompute_steps(self) -> None:
        # Dummy method for compatibility; steps are computed on-the-fly.
        return


@dataclass
class CacheConfig:
    """
    Configuration for cached DRP image stacks.
    """

    ph_slice: int = 1
    th_slice: int = 1
    data_serial: str | int | None = None


def load_drp_config(path: Path) -> DRPConfig:
    """
    Load DRP acquisition configuration from a YAML file.
    
    :param path: Path to the YAML configuration file.
    :type path: Path
    :return: DRPConfig instance with loaded settings.
    :rtype: DRPConfig
    """

    path = Path(path)
    with path.open("r") as fh:
        data = yaml.safe_load(fh) or {}

    # Check for must-exist keys
    required_keys = ["th_min", "th_max", "th_num", "ph_min", "ph_max", "ph_num"]
    missing = [k for k in required_keys if k not in data]
    if missing:
        raise ValueError(f"Missing required keys in DRP config: {missing}")

    cfg = DRPConfig(
        th_min=data["th_min"],
        th_max=data["th_max"],
        th_num=data["th_num"],
        ph_min=data["ph_min"],
        ph_max=data["ph_max"],
        ph_num=data["ph_num"],
        phi_slice=data.get("phi_slice", 1),
        theta_slice=data.get("theta_slice", 1),
        data_serial=data.get("data_serial"),
    )
    cfg.validate() # Check for validity
    return cfg


def load_cache_config(path: Path) -> CacheConfig:
    path = Path(path)
    if not path.exists():
        return CacheConfig()
    with path.open("r") as fh:
        data = yaml.safe_load(fh) or {}
    return CacheConfig(
        ph_slice=data.get("ph_slice", 1),
        th_slice=data.get("th_slice", 1),
        data_serial=data.get("data_serial"),
    )


def save_cache_config(path: Path, cfg: CacheConfig) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        yaml.dump(asdict(cfg), fh)


def save_drp_config(path: Path, cfg: DRPConfig) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        yaml.dump(asdict(cfg), fh, sort_keys=False)


@dataclass
class Settings:
    """
    Centralised configuration for PaperDRM.

    This bundles DRP acquisition parameters with runtime knobs used to load and
    process images. Use ``Settings.from_yaml`` to hydrate from ``exp_param.yaml``
    (or another config) and optionally override fields with ``with_overrides``.
    """

    data_root: str | Path = "data"
    folder: str | Path | None = None
    img_format: str = "jpg"
    angle_slice: tuple[int, int] = (1, 1)
    use_cached_stack: bool = True
    subtract_background: bool = True
    subtraction_scale_percentile: float = 99.5
    load_workers: int | None = None
    config_path: str | Path | None = None
    drp: DRPConfig | None = None
    square_crop: bool = False
    theta_min_deg: float | None = None
    fov_width_cm: float | None = None
    verbose: bool = False

    def __post_init__(self) -> None:
        self.data_root = Path(self.data_root)
        self.folder = Path(self.folder) if self.folder is not None else None
        self.config_path = Path(self.config_path) if self.config_path is not None else None
        self.angle_slice = tuple(self.angle_slice)  # type: ignore[assignment]
        self.validate()

    @property
    def data_serial(self) -> str | int | None:
        return self.drp.data_serial if self.drp else None

    def validate(self) -> None:
        if self.drp is None:
            raise ValueError("Settings.drp must be provided (load via Settings.from_yaml or supply DRPConfig).")
        self.drp.validate()
        if self.theta_min_deg is not None and float(self.theta_min_deg) > float(self.drp.th_max):
            raise ValueError("theta_min_deg cannot exceed DRP th_max.")
        if self.fov_width_cm is not None and float(self.fov_width_cm) <= 0:
            raise ValueError("fov_width_cm must be positive when provided.")
        if len(self.angle_slice) != 2:
            raise ValueError("angle_slice must be a 2-tuple of (phi_slice, theta_slice).")
        ph_slice, th_slice = self.angle_slice
        if ph_slice <= 0 or th_slice <= 0:
            raise ValueError("angle_slice values must be positive.")
        if self.drp.ph_num % ph_slice != 0 or self.drp.th_num % th_slice != 0:
            raise ValueError("angle_slice must evenly divide the DRP phi/theta counts.")

    def with_overrides(self, **kwargs: Any) -> "Settings":
        """
        Return a new Settings with select fields replaced.
        """
        return replace(self, **kwargs)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Settings":
        """
        Load settings from a YAML file. DRP parameters are required; other
        runtime fields are optional and fall back to defaults.
        """
        cfg_path = Path(path)
        raw: dict[str, Any] = yaml.safe_load(cfg_path.read_text()) or {}
        drp_cfg = load_drp_config(cfg_path)

        angle_slice = tuple(raw.get("angle_slice", (1, 1)))
        return cls(
            data_root=raw.get("data_root", "data"),
            folder=raw.get("folder"),
            img_format=raw.get("img_format", "jpg"),
            angle_slice=angle_slice,  # type: ignore[arg-type]
            use_cached_stack=raw.get("use_cached_stack", True),
            subtract_background=raw.get("subtract_background", True),
            subtraction_scale_percentile=raw.get("subtraction_scale_percentile", 99.5),
            load_workers=raw.get("load_workers"),
            config_path=cfg_path,
            drp=drp_cfg,
            square_crop=raw.get("square_crop", False),
            theta_min_deg=raw.get("theta_min_deg"),
            fov_width_cm=raw.get("fov_width_cm"),
            verbose=raw.get("verbose", False),
        )
