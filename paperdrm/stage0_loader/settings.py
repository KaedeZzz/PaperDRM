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


_ACQ_KEYS = ("th_min", "th_max", "th_num", "ph_min", "ph_max", "ph_num")


def resolve_drp_from_yaml(path: Path) -> tuple[DRPConfig | None, str | int | None]:
    """
    Decide whether a yaml file fully specifies the DRP acquisition grid.

    Returns ``(drp_cfg, data_serial_hint)`` where exactly one of the two is
    informative: a fully-specified yaml yields ``(DRPConfig, None)``; an empty
    or acq-free yaml yields ``(None, raw_data_serial)`` so the caller can
    populate the grid via inference later. A partial acq spec raises.
    """
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    present = [k for k in _ACQ_KEYS if k in raw]
    if 0 < len(present) < len(_ACQ_KEYS):
        missing = [k for k in _ACQ_KEYS if k not in raw]
        raise ValueError(
            f"Partial DRP acquisition fields in {path}: missing {missing}. "
            "Provide all six (th_min/max/num, ph_min/max/num) or omit them all "
            "to enable inference from filenames."
        )
    if len(present) == len(_ACQ_KEYS):
        return load_drp_config(path), None
    return None, raw.get("data_serial")


@dataclass
class Settings:
    """
    Centralised configuration for PaperDRM.

    This bundles DRP acquisition parameters with runtime knobs used to load and
    process images. Use ``Settings.from_yaml`` to hydrate from ``exp_param.yaml``
    (or another config) and optionally override fields with ``with_overrides``.

    ``drp`` may be left as None when the yaml omits the six acquisition fields
    (th_min/max/num, ph_min/max/num); ``ImagePack`` will then populate it by
    inferring the grid from the image folder. ``data_serial_hint`` carries the
    yaml-supplied serial in that case until inference attaches it.
    """

    data_root: str | Path = "data"
    folder: str | Path | None = None
    image_path: str | Path | None = None
    img_format: str = "jpg"
    angle_slice: tuple[int, int] = (1, 1)
    use_cached_stack: bool = True
    subtract_background: bool = True
    subtraction_scale_percentile: float = 99.5
    load_workers: int | None = None
    config_path: str | Path | None = None
    drp: DRPConfig | None = None
    data_serial_hint: str | int | None = None
    square_crop: bool = False
    theta_min_deg: float | None = None
    fov_width_cm: float | None = None
    crop_roi: tuple[int, int, int, int] | None = None
    period_range_cm: tuple[float, float] | None = None
    line_dir_deg: float = 90.0
    auto_line_dir: bool = False
    wire_is_darker: bool = True
    verbose: bool = False

    def __post_init__(self) -> None:
        self.data_root = Path(self.data_root)
        self.folder = Path(self.folder) if self.folder is not None else None
        self.image_path = Path(self.image_path) if self.image_path is not None else None
        self.config_path = Path(self.config_path) if self.config_path is not None else None
        self.angle_slice = tuple(self.angle_slice)  # type: ignore[assignment]
        if self.crop_roi is not None:
            self.crop_roi = tuple(int(v) for v in self.crop_roi)  # type: ignore[assignment]
        if self.period_range_cm is not None:
            self.period_range_cm = tuple(float(v) for v in self.period_range_cm)  # type: ignore[assignment]
        self.validate()

    @property
    def data_serial(self) -> str | int | None:
        if self.drp is not None and self.drp.data_serial is not None:
            return self.drp.data_serial
        return self.data_serial_hint

    def validate(self) -> None:
        if len(self.angle_slice) != 2:
            raise ValueError("angle_slice must be a 2-tuple of (phi_slice, theta_slice).")
        ph_slice, th_slice = self.angle_slice
        if ph_slice <= 0 or th_slice <= 0:
            raise ValueError("angle_slice values must be positive.")
        if self.fov_width_cm is not None and float(self.fov_width_cm) <= 0:
            raise ValueError("fov_width_cm must be positive when provided.")
        if self.period_range_cm is not None:
            if len(self.period_range_cm) != 2 or self.period_range_cm[0] >= self.period_range_cm[1]:
                raise ValueError("period_range_cm must be (lo, hi) with lo < hi.")
            if self.period_range_cm[0] <= 0:
                raise ValueError("period_range_cm values must be positive.")
        if self.crop_roi is not None:
            if len(self.crop_roi) != 4:
                raise ValueError("crop_roi must be [x, y, w, h].")
            x, y, w, h = self.crop_roi
            if w <= 0 or h <= 0:
                raise ValueError("crop_roi w and h must be positive.")
        if self.drp is not None:
            self.drp.validate()
            if self.theta_min_deg is not None and float(self.theta_min_deg) > float(self.drp.th_max):
                raise ValueError("theta_min_deg cannot exceed DRP th_max.")
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
        Load settings from a YAML file.

        The six DRP acquisition fields (th_min/max/num, ph_min/max/num) are
        treated as all-or-nothing: provide all six to fix the grid explicitly,
        or omit all six to let ``ImagePack`` infer them from filenames. A
        partial set raises immediately so users notice the typo.
        """
        cfg_path = Path(path)
        raw: dict[str, Any] = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}

        drp_cfg, data_serial_hint = resolve_drp_from_yaml(cfg_path)

        angle_slice = tuple(raw.get("angle_slice", (1, 1)))
        _serial = raw.get("data_serial")
        _data_root = raw.get("data_root") or (
            Path("data") / "drp" / str(_serial) if _serial is not None else Path("data")
        )
        return cls(
            data_root=_data_root,
            folder=raw.get("folder"),
            image_path=raw.get("image_path"),
            img_format=raw.get("img_format", "jpg"),
            angle_slice=angle_slice,  # type: ignore[arg-type]
            use_cached_stack=raw.get("use_cached_stack", True),
            subtract_background=raw.get("subtract_background", True),
            subtraction_scale_percentile=raw.get("subtraction_scale_percentile", 99.5),
            load_workers=raw.get("load_workers"),
            config_path=cfg_path,
            drp=drp_cfg,
            data_serial_hint=data_serial_hint,
            square_crop=raw.get("square_crop", False),
            theta_min_deg=raw.get("theta_min_deg"),
            fov_width_cm=raw.get("fov_width_cm"),
            crop_roi=tuple(raw["crop_roi"]) if raw.get("crop_roi") else None,
            period_range_cm=tuple(raw["period_range_cm"]) if raw.get("period_range_cm") else None,  # type: ignore[arg-type]
            line_dir_deg=raw.get("line_dir_deg", 90.0),
            auto_line_dir=bool(raw.get("auto_line_dir", False)),
            wire_is_darker=bool(raw.get("wire_is_darker", True)),
            verbose=raw.get("verbose", False),
        )
