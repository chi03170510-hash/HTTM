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
    strength: float = 0.45,
    bilateral_d: int = 5,
) -> np.ndarray:
    """Apply bilateral denoising followed by unsharp masking: I_sharp = (1+a)*I - a*(G*I)."""
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
    blurred = cv2.GaussianBlur(base, (0, 0), sigmaX=max(0.5, float(sigma)))
    return cv2.addWeighted(base, 1.0 + float(strength), blurred, -float(strength), 0)


def preprocess_frame(frame_bgr: np.ndarray, config: dict[str, Any] | None = None) -> np.ndarray:
    """Run configured spatial-domain frame enhancement before detector inference."""
    if not isinstance(frame_bgr, np.ndarray) or frame_bgr.size == 0:
        return frame_bgr
    cfg = config or {}
    out = frame_bgr
    if cfg.get("preprocess_auto_gamma_enabled", True):
        out = auto_gamma_correction(out)
    if cfg.get("preprocess_clahe_enabled", True):
        clip = float(cfg.get("preprocess_clahe_clip_limit", 1.8))
        grid = int(cfg.get("preprocess_clahe_tile_grid", 8))
        out = apply_clahe_lab(out, clip_limit=clip, tile_grid_size=(grid, grid))
    return out
