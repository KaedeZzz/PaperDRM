"""Load only the images required by the active native detector track."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import numpy as np

from paperdrm.config import AcquisitionConfig, PipelineConfig
from paperdrm.models import DetectorTrack
from paperdrm.pipeline import MultiPhiInput, PipelineInput, SingleImageInput
from paperdrm.stage0_loader.inference import (
    infer_drp_config_from_folder,
    verify_drp_match,
)
from paperdrm.stage0_loader.settings import DRPConfig


_ANGLE_FILENAME = re.compile(r"^(?P<phi>\d+)_(?P<theta>\d+)\.[^.]+$")


@dataclass(frozen=True)
class PreparedInput:
    config: PipelineConfig
    input_data: PipelineInput
    input_paths: tuple[Path, ...]
    display_images: tuple[np.ndarray, ...]


def _read_grayscale(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise OSError(f"could not open grayscale image: {path}")
    return image


def _validate_crop(
    crop: tuple[int, int, int, int],
    shape: tuple[int, ...],
) -> None:
    x, y, width, height = crop
    image_height, image_width = shape[:2]
    if x < 0 or y < 0 or x + width > image_width or y + height > image_height:
        raise ValueError(
            f"crop_roi {crop} is outside image bounds {image_width}x{image_height}"
        )


def _crop_and_square(
    image: np.ndarray,
    config: PipelineConfig,
) -> np.ndarray:
    output = image
    if config.crop_roi is not None:
        _validate_crop(config.crop_roi, output.shape)
        x, y, width, height = config.crop_roi
        output = output[y : y + height, x : x + width]
    if config.square_crop:
        height, width = output.shape[:2]
        size = min(height, width)
        y = (height - size) // 2
        x = (width - size) // 2
        output = output[y : y + size, x : x + size]
    return output


def _subtract_background(
    image: np.ndarray,
    config: PipelineConfig,
    background: np.ndarray | None,
) -> np.ndarray:
    if not config.subtract_background:
        return image.copy()
    if background is None:
        background = cv2.GaussianBlur(
            image,
            (0, 0),
            sigmaX=100,
            borderType=cv2.BORDER_REFLECT_101,
        )
    if background.shape != image.shape:
        raise ValueError(
            f"background shape {background.shape} does not match image shape {image.shape}"
        )
    difference = image.astype(np.float32) - background.astype(np.float32)
    difference = np.clip(difference, 0, None)
    reference = float(
        np.percentile(difference, config.subtraction_scale_percentile)
    )
    scale = 255.0 / max(reference, 1.0)
    return np.clip(difference * scale, 0, 255).astype(np.uint8)


def _effective_config(
    config: PipelineConfig,
    *,
    original_width: int,
    output_width: int,
) -> PipelineConfig:
    if config.fov_width_cm is None or original_width == output_width:
        return config
    return replace(
        config,
        fov_width_cm=config.fov_width_cm * output_width / original_width,
    )


def _acquisition_to_drp(config: AcquisitionConfig) -> DRPConfig:
    return DRPConfig(
        th_min=config.theta_min,
        th_max=config.theta_max,
        th_num=config.theta_count,
        ph_min=config.phi_min,
        ph_max=config.phi_max,
        ph_num=config.phi_count,
    )


def _resolve_single_image(config: PipelineConfig) -> Path:
    assert config.image_path is not None
    raw = config.image_path
    candidates = [raw]
    if not raw.is_absolute() and config.config_path is not None:
        candidates.append(config.config_path.parent / raw)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"single image does not exist: {raw}")


def _resolve_drp_folder(config: PipelineConfig) -> Path:
    if config.folder is None:
        folder = config.data_root / "raw"
    elif config.folder.is_absolute():
        folder = config.folder
    else:
        folder = config.data_root / config.folder
    if not folder.is_dir():
        raise FileNotFoundError(f"DRP image folder does not exist: {folder}")
    return folder.resolve()


def _angle_paths(folder: Path, image_format: str) -> dict[tuple[int, int], Path]:
    values: dict[tuple[int, int], Path] = {}
    for path in folder.glob(f"*.{image_format}"):
        match = _ANGLE_FILENAME.fullmatch(path.name)
        if match is None:
            continue
        angle = (int(match.group("phi")), int(match.group("theta")))
        if angle in values:
            raise ValueError(f"duplicate DRP angle file for {angle}: {path}")
        values[angle] = path.resolve()
    if not values:
        raise ValueError(
            f"no filenames matched <phi>_<theta>.{image_format} in {folder}"
        )
    return values


class FilesystemInputProvider:
    """Prepare native V2 input without writing cache or result files."""

    def prepare(self, config: PipelineConfig) -> PreparedInput:
        if config.track is DetectorTrack.SINGLE_IMAGE:
            return self._single_image(config)
        if config.track in (DetectorTrack.SIMPLE, DetectorTrack.MULTI_PHI):
            return self._drp_grazing_images(config)
        raise NotImplementedError("legacy input remains on the V1 compatibility path")

    def _single_image(self, config: PipelineConfig) -> PreparedInput:
        path = _resolve_single_image(config)
        raw = _read_grayscale(path)
        processed = _subtract_background(raw, config, background=None)
        display = _crop_and_square(raw, config)
        processed = _crop_and_square(processed, config)
        effective = _effective_config(
            config,
            original_width=raw.shape[1],
            output_width=processed.shape[1],
        )
        return PreparedInput(
            config=effective,
            input_data=SingleImageInput(processed),
            input_paths=(path,),
            display_images=(display,),
        )

    def _drp_grazing_images(self, config: PipelineConfig) -> PreparedInput:
        folder = _resolve_drp_folder(config)
        inferred, _ = infer_drp_config_from_folder(
            folder,
            img_format=config.image_format,
            strict=True,
        )
        if config.acquisition is not None:
            acquisition = _acquisition_to_drp(config.acquisition)
            verify_drp_match(acquisition, inferred, source=folder)
        else:
            acquisition = inferred

        phi_values = list(
            range(
                int(acquisition.ph_min),
                int(acquisition.ph_max) + 1,
                int(round(acquisition.ph_step)),
            )
        )
        theta_values = list(
            range(
                int(acquisition.th_min),
                int(acquisition.th_max) + 1,
                int(round(acquisition.th_step)),
            )
        )
        phi_slice, theta_slice = config.angle_slice
        if len(phi_values) % phi_slice or len(theta_values) % theta_slice:
            raise ValueError("angle_slice must evenly divide phi and theta counts")
        phi_values = phi_values[::phi_slice]
        theta_values = theta_values[::theta_slice]
        if config.theta_min_deg is not None:
            theta_values = [
                value for value in theta_values if value >= config.theta_min_deg
            ]
        if not theta_values:
            raise ValueError("theta_min_deg removed every theta sample")

        paths_by_angle = _angle_paths(folder, config.image_format)
        grazing_theta = theta_values[-1]
        selected_paths = tuple(
            paths_by_angle[(phi, grazing_theta)] for phi in phi_values
        )
        selected_phi = phi_values
        if config.track is DetectorTrack.SIMPLE:
            selected_paths = (selected_paths[0],)
            selected_phi = phi_values[:1]
        elif len(selected_paths) < 2:
            raise ValueError("multi_phi requires at least two selected phi images")

        background_folder = None
        if config.subtract_background:
            for candidate in (
                folder / "background",
                config.data_root / "background",
            ):
                if candidate.is_dir() and all(
                    (candidate / path.name).is_file() for path in selected_paths
                ):
                    background_folder = candidate
                    break

        raw_images = tuple(_read_grayscale(path) for path in selected_paths)
        if any(image.shape != raw_images[0].shape for image in raw_images):
            raise ValueError("selected grazing images must have identical shapes")
        processed_images = []
        display_images = []
        for path, raw in zip(selected_paths, raw_images):
            background = (
                _read_grayscale(background_folder / path.name)
                if background_folder is not None
                else None
            )
            processed = _subtract_background(raw, config, background)
            processed_images.append(_crop_and_square(processed, config))
            display_images.append(_crop_and_square(raw, config))

        effective = _effective_config(
            config,
            original_width=raw_images[0].shape[1],
            output_width=processed_images[0].shape[1],
        )
        if config.track is DetectorTrack.SIMPLE:
            input_data: PipelineInput = SingleImageInput(processed_images[0])
            inputs = selected_paths
            displays = (display_images[0],)
        else:
            input_data = MultiPhiInput(
                tuple(processed_images),
                phi_deg=tuple(float(value) for value in selected_phi),
            )
            inputs = selected_paths
            displays = tuple(display_images)
        return PreparedInput(
            config=effective,
            input_data=input_data,
            input_paths=inputs,
            display_images=displays,
        )
