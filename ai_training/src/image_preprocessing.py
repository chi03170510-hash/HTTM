"""Spatial-domain image preprocessing for classroom/webcam behavior & face pipelines.

Implements classical Image Processing operations before deep-learning inference:
1. Adaptive luminance contrast enhancement via CLAHE in CIE LAB color space.
2. Automatic LUT-based gamma correction for backlit / low-light webcam frames.
3. Edge-preserving bilateral denoising + Unsharp Masking for upscaled ROI crops.
"""
from __future__ import annotations

from typing import Any
import cv2
import numpy as np


def apply_clahe_lab(
    frame_bgr: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Enhance local contrast on the L channel in LAB space without shifting hue."""
    if (
        not isinstance(frame_bgr, np.ndarray)
        or frame_bgr.ndim != 3
        or frame_bgr.shape[2] != 3
        or frame_bgr.dtype != np.uint8
        or frame_bgr.size == 0
    ):
        return frame_bgr
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)
    grid_w = max(2, int(tile_grid_size[0]))
    grid_h = max(2, int(tile_grid_size[1]))
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(grid_w, grid_h))
    l_enhanced = clahe.apply(l_chan)
    merged = cv2.merge((l_enhanced, a_chan, b_chan))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def auto_gamma_correction(
    frame_bgr: np.ndarray,
    target_mean: float = 122.0,
    min_gamma: float = 0.65,
    max_gamma: float = 1.45,
) -> np.ndarray:
    """Apply LUT gamma correction when mean luminance deviates from target range."""
    if (
        not isinstance(frame_bgr, np.ndarray)
        or frame_bgr.ndim != 3
        or frame_bgr.shape[2] != 3
        or frame_bgr.dtype != np.uint8
        or frame_bgr.size == 0
    ):
        return frame_bgr
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    mean_l = float(np.mean(gray))
    if 95.0 <= mean_l <= 155.0:
        return frame_bgr
    gamma = np.clip(np.log(target_mean / 255.0) / np.log(max(mean_l, 1.0) / 255.0), min_gamma, max_gamma)
    if abs(gamma - 1.0) < 0.05:
        return frame_bgr
    lut = np.array(
        [np.clip(((i / 255.0) ** gamma) * 255.0, 0, 255) for i in range(256)],
        dtype=np.uint8,
    )
    return cv2.LUT(frame_bgr, lut)


def denoise_and_sharpen_roi(
    crop_bgr: np.ndarray,
    sigma: float = 1.0,
    strength: float = 0.20,
    bilateral_d: int = 5,
) -> np.ndarray:
    """Apply bilateral denoising followed by gentle conditional unsharp masking."""
    if (
        not isinstance(crop_bgr, np.ndarray)
        or crop_bgr.ndim != 3
        or crop_bgr.shape[2] != 3
        or crop_bgr.dtype != np.uint8
        or crop_bgr.size == 0
    ):
        return crop_bgr
    if bilateral_d > 0 and min(crop_bgr.shape[:2]) >= 16:
        base = cv2.bilateralFilter(crop_bgr, d=int(bilateral_d), sigmaColor=25, sigmaSpace=25)
    else:
        base = crop_bgr
    if strength <= 0.0:
        return base
    gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    # Skip sharpening if the crop already has strong high-frequency edges/noise
    if lap_var > 180.0:
        return base
    eff_strength = min(float(strength), 0.25)
    blurred = cv2.GaussianBlur(base, (0, 0), sigmaX=max(0.5, float(sigma)))
    return cv2.addWeighted(base, 1.0 + eff_strength, blurred, -eff_strength, 0)


def preprocess_frame(frame_bgr: np.ndarray, config: dict[str, Any] | None = None) -> np.ndarray:
    """Run adaptive spatial-domain frame enhancement only on low-light/low-contrast frames.

    Avoids unconditional CLAHE/gamma on well-lit frames, which amplifies background
    sensor noise and causes false-positive YOLO proposals.
    """
    if (
        not isinstance(frame_bgr, np.ndarray)
        or frame_bgr.ndim != 3
        or frame_bgr.shape[2] != 3
        or frame_bgr.dtype != np.uint8
        or frame_bgr.size == 0
    ):
        return frame_bgr
    cfg = config or {}
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    mean_l = float(np.mean(gray))
    std_l = float(np.std(gray))
    low_light_max = float(cfg.get("preprocess_low_light_mean_max", 85.0))
    high_light_min = float(cfg.get("preprocess_high_light_mean_min", 175.0))
    low_contrast_max = float(cfg.get("preprocess_low_contrast_std_max", 34.0))

    needs_gamma = mean_l < low_light_max or mean_l > high_light_min
    needs_clahe = mean_l < low_light_max or std_l < low_contrast_max
    if not needs_gamma and not needs_clahe:
        return frame_bgr

    out = frame_bgr
    # Lightly denoise very dark frames before boosting contrast so sensor noise is not amplified
    if mean_l < 65.0 and min(out.shape[:2]) >= 32:
        out = cv2.GaussianBlur(out, (3, 3), sigmaX=0.6)
    if cfg.get("preprocess_auto_gamma_enabled", True) and needs_gamma:
        out = auto_gamma_correction(out)
    if cfg.get("preprocess_clahe_enabled", True) and needs_clahe:
        clip = float(cfg.get("preprocess_clahe_clip_limit", 1.5))
        grid = int(cfg.get("preprocess_clahe_tile_grid", 8))
        out = apply_clahe_lab(out, clip_limit=clip, tile_grid_size=(grid, grid))
    return out
