"""
TinyUNet Motion Detection — чистий inference-скрипт (без калібрування і без Core ML).

Джерело: витягнуто з ноутбука `best__15_.ipynb` (production candidate,
hard_negative_round01). Тут залишено рівно те, що потрібно для запуску
inference на відео:

    відео -> CompensationPipeline (класична компенсація руху камери,
             3 вирівняних grayscale кадри) -> TinyUNet (нейромережа,
             ймовірність руху) -> контурний фільтр -> SORT tracker
             -> bbox по кадрах -> (опційно) відео з відмальованими bbox

Додатково є два інструменти профілювання:
    1. count_model_flops()   — точний підрахунок FLOPs/MACs TinyUNet
                                (лише для нейромережі; в класичній
                                частині немає "FLOPs" у класичному
                                розумінні — там оптичний потік,
                                RANSAC, варп зображень тощо).
    2. benchmark_pipeline()  — вимірює реальну швидкість (ms/frame, FPS)
                                окремо для класики і окремо для
                                нейромережі, плюс сумарну пропускну
                                здатність усього пайплайна.

Запуск:
    python motion_inference.py --video path/to/video.mp4 \
        --checkpoint path/to/best_unet.pth \
        --output out/result.mp4

    # тільки профілювання швидкості/FLOPs, без збереження відео:
    python motion_inference.py --video path/to/video.mp4 \
        --checkpoint path/to/best_unet.pth \
        --benchmark --max-frames 300
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import json
import queue
import re
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Dict, Iterator, List, Mapping, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


# ================================================================
# DEVICE / GLOBAL SETTINGS
# ================================================================

if torch.cuda.is_available():
    DEVICE = "cuda"
elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

PROCESSING_MAX_WIDTH = 640
FUSE_BATCHNORM_FOR_INFERENCE = True
USE_PREALLOCATED_UNET_INPUT = True

# Скільки потоків дозволено внутрішньому пулу OpenCV.
#   1 — рекомендовано для конвеєра: producer тримає одне ядро,
#       consumer не конкурує з пулом OpenCV за решту;
#   0 — "стільки, скільки вирішить OpenCV" (дефолт бібліотеки);
#   None — не чіпати глобальний стан.
OPENCV_THREADS: Optional[int] = 1

# Скільки ітерацій прогріву робить нейробекенд перед вимірюванням.
# Перший inference на MPS — компіляція Metal-шейдерів, на Core ML —
# компіляція графа і перенос на ANE. Це сотні мілісекунд, які не мають
# потрапити в статистику.
NEURAL_WARMUP_ITERATIONS = 30

# Глибина черги між producer (класика, CPU) і consumer (мережа, GPU/ANE).
# Обмежена черга дає backpressure: producer блокується замість того,
# щоб нескінченно нарощувати latency і пам'ять.
PIPELINE_QUEUE_SIZE = 3


@contextlib.contextmanager
def configure_opencv_threads(num_threads: Optional[int]):
    """
    Тимчасово виставляє кількість потоків OpenCV і ГАРАНТОВАНО відновлює її.

    Повертає dict з `requested` / `actual` / `previous`, щоб у звіті було
    видно, чи бібліотека справді послухалась (у старих збірках
    ``getNumThreads()`` може повернути не те, що просили).
    """
    previous = int(cv2.getNumThreads())
    info = {"requested": num_threads, "previous": previous, "actual": previous}
    if num_threads is not None:
        cv2.setNumThreads(int(num_threads))
        info["actual"] = int(cv2.getNumThreads())
    try:
        yield info
    finally:
        cv2.setNumThreads(previous)
        info["restored_to"] = int(cv2.getNumThreads())


# ================================================================
# 1. PRODUCTION CAMERA-MOTION COMPENSATION (класична частина)
# ================================================================

PRODUCTION_CONFIG = {
    "max_width": PROCESSING_MAX_WIDTH,
    "redetect_interval": 8,
    "request_decoder_scaling": True,
    "preprocess_gray_first": True,

    "grid_rows": 10,
    "grid_cols": 10,
    "points_per_cell": 3,
    "quality_level": 0.01,
    "min_distance": 7,
    "block_size": 7,

    "lk_win_size": 11,
    "lk_max_level": 2,
    "min_eig_threshold": 1e-4,
    "fb_threshold": 0.75,
    "min_correspondences": 50,

    "motion_model": "homography",
    "ransac_threshold": 2.0,
    "min_ransac_inliers": 25,
    "min_inlier_ratio": 0.50,
    "low_confidence_inlier_ratio": 0.65,
    "min_model_scale": 0.50,
    "max_model_scale": 2.00,
}

_REQUIRED_CONFIG_KEYS = frozenset(PRODUCTION_CONFIG)
_EMPTY_TENSOR = np.empty((0, 0, 0), dtype=np.float32)
_IMAGE_SUFFIXES = frozenset({".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"})


@dataclass(frozen=True)
class CompensatedSample:
    """Один network input, вирівняний на ``frame_index``.

    ``tensor`` має форму ``(4, H, W)`` для валідних семплів. Для невалідних —
    порожній float32 масив; перед inference потрібно перевіряти ``valid``.
    """

    tensor: np.ndarray
    frame_index: int
    valid: bool
    inlier_ratio: float
    skip_reason: Optional[str]


@dataclass(frozen=True)
class _MotionCacheEntry:
    valid: bool
    matrix: Optional[np.ndarray]
    inlier_ratio: float
    failure_reason: Optional[str]


def _validate_config(config: Mapping[str, object]) -> dict:
    missing = sorted(_REQUIRED_CONFIG_KEYS.difference(config))
    if missing:
        raise KeyError(f"Missing production config keys: {', '.join(missing)}")

    validated = {key: config[key] for key in PRODUCTION_CONFIG}

    positive_integer_keys = (
        "max_width", "grid_rows", "grid_cols", "points_per_cell", "block_size",
        "lk_win_size", "min_correspondences", "min_ransac_inliers", "redetect_interval",
    )
    for key in positive_integer_keys:
        if int(validated[key]) <= 0:
            raise ValueError(f"config[{key!r}] must be positive")

    if int(validated["lk_max_level"]) < 0:
        raise ValueError("config['lk_max_level'] must be non-negative")

    positive_float_keys = (
        "quality_level", "min_distance", "min_eig_threshold", "fb_threshold",
        "ransac_threshold", "min_model_scale", "max_model_scale",
    )
    for key in positive_float_keys:
        if float(validated[key]) <= 0:
            raise ValueError(f"config[{key!r}] must be positive")

    for key in ("min_inlier_ratio", "low_confidence_inlier_ratio"):
        value = float(validated[key])
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"config[{key!r}] must be in [0, 1]")

    if float(validated["min_model_scale"]) > float(validated["max_model_scale"]):
        raise ValueError("min_model_scale must not exceed max_model_scale")

    if str(validated["motion_model"]) not in {"affine_partial", "affine_full", "homography"}:
        raise ValueError("motion_model must be affine_partial, affine_full or homography")

    return validated


def _align_dimension_to_8(value: int, *, round_up: bool) -> int:
    """Повертає додатний розмір, кратний 8."""
    value = max(1, int(value))
    if round_up:
        return max(8, ((value + 7) // 8) * 8)
    return max(8, ((value + 4) // 8) * 8)


def _scaled_size(width: int, height: int, max_width: int) -> Tuple[Tuple[int, int], float]:
    if int(width) <= 0 or int(height) <= 0:
        raise ValueError("width and height must be positive")
    if int(max_width) <= 0:
        raise ValueError("max_width must be positive")

    scale = min(1.0, float(max_width) / float(width))
    raw_width = max(1, int(round(width * scale)))
    raw_height = max(1, int(round(height * scale)))

    # TinyUNet має три рівні downsampling, тому обидва розміри
    # повинні ділитися на 8. Ширину округлюємо до найближчого
    # допустимого значення, висоту — вгору, щоб не втрачати пікселі.
    max_aligned_width = max(8, (int(max_width) // 8) * 8)
    output_width = min(
        max_aligned_width,
        _align_dimension_to_8(raw_width, round_up=False),
    )
    output_height = _align_dimension_to_8(raw_height, round_up=True)

    return (output_width, output_height), scale


def preprocess_frame(frame: np.ndarray, max_width: int, gray_first: bool = True) -> Tuple[np.ndarray, float]:
    if not isinstance(frame, np.ndarray) or frame.size == 0:
        raise ValueError("frame must be a non-empty numpy array")
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("frame must be a BGR image with shape H×W×3")
    if frame.dtype != np.uint8:
        raise ValueError("frame must have dtype uint8")
    if max_width is None or int(max_width) <= 0:
        raise ValueError("max_width must be positive")

    height, width = frame.shape[:2]
    output_size, scale = _scaled_size(width, height, int(max_width))
    needs_resize = (width, height) != output_size

    if bool(gray_first):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if needs_resize:
            gray = cv2.resize(gray, output_size, interpolation=cv2.INTER_AREA)
    else:
        resized = (
            cv2.resize(frame, output_size, interpolation=cv2.INTER_AREA)
            if needs_resize else frame
        )
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    return np.ascontiguousarray(gray), scale


def _request_decoder_scaling(capture: cv2.VideoCapture, max_width: int) -> Optional[Tuple[int, int]]:
    source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if source_width <= 0 or source_height <= 0:
        return None

    target_size, _ = _scaled_size(source_width, source_height, int(max_width))
    if target_size != (source_width, source_height):
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, int(target_size[0]))
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, int(target_size[1]))
    return target_size

def detect_points_grid(gray: np.ndarray, config: Mapping[str, object]) -> np.ndarray:
    if not isinstance(gray, np.ndarray) or gray.ndim != 2 or gray.dtype != np.uint8:
        raise ValueError("gray must be a non-empty uint8 image with shape H×W")
    if gray.size == 0:
        raise ValueError("gray must not be empty")

    grid_rows = int(config["grid_rows"])
    grid_cols = int(config["grid_cols"])
    points_per_cell = int(config["points_per_cell"])
    min_distance = float(config["min_distance"])
    block_size = int(config["block_size"])
    if min(grid_rows, grid_cols, points_per_cell, block_size) <= 0:
        raise ValueError("grid dimensions, points_per_cell, and block_size must be positive")

    height, width = gray.shape
    x_edges = np.linspace(0, width, grid_cols + 1, dtype=np.int32)
    y_edges = np.linspace(0, height, grid_rows + 1, dtype=np.int32)
    candidates = []

    for row in range(grid_rows):
        y0, y1 = int(y_edges[row]), int(y_edges[row + 1])
        for col in range(grid_cols):
            x0, x1 = int(x_edges[col]), int(x_edges[col + 1])
            cell = gray[y0:y1, x0:x1]
            if cell.shape[0] < block_size or cell.shape[1] < block_size:
                continue

            local_points = cv2.goodFeaturesToTrack(
                cell, maxCorners=points_per_cell,
                qualityLevel=float(config["quality_level"]),
                minDistance=min_distance, blockSize=block_size,
            )
            if local_points is None:
                continue

            local_points = local_points.reshape(-1, 2).astype(np.float32)
            local_points[:, 0] += x0
            local_points[:, 1] += y0
            candidates.extend(local_points)

    if not candidates:
        return np.empty((0, 1, 2), dtype=np.float32)

    accepted = []
    min_distance_squared = min_distance ** 2
    bucket_size = max(min_distance, 1e-6)
    buckets: Dict[Tuple[int, int], list] = {}

    def bucket_key(x: float, y: float) -> Tuple[int, int]:
        return (int(np.floor(x / bucket_size)), int(np.floor(y / bucket_size)))

    for point in np.asarray(candidates, dtype=np.float32):
        if not np.isfinite(point).all():
            continue

        bucket_x, bucket_y = bucket_key(float(point[0]), float(point[1]))
        too_close = False
        for delta_x in (-1, 0, 1):
            if too_close:
                break
            for delta_y in (-1, 0, 1):
                neighbour_indices = buckets.get((bucket_x + delta_x, bucket_y + delta_y))
                if not neighbour_indices:
                    continue
                for other_index in neighbour_indices:
                    other_point = accepted[other_index]
                    diff_x = float(other_point[0] - point[0])
                    diff_y = float(other_point[1] - point[1])
                    if diff_x * diff_x + diff_y * diff_y < min_distance_squared:
                        too_close = True
                        break
                if too_close:
                    break
        if too_close:
            continue

        buckets.setdefault((bucket_x, bucket_y), []).append(len(accepted))
        accepted.append(point)

    if not accepted:
        return np.empty((0, 1, 2), dtype=np.float32)
    return np.asarray(accepted, dtype=np.float32).reshape(-1, 1, 2)


def _merge_spaced_points(existing_points: np.ndarray, candidate_points: np.ndarray, min_distance: float) -> np.ndarray:
    existing = np.asarray(existing_points, dtype=np.float32).reshape(-1, 2)
    candidates = np.asarray(candidate_points, dtype=np.float32).reshape(-1, 2)

    if len(existing) == 0:
        return candidates.reshape(-1, 1, 2)
    if len(candidates) == 0:
        return existing.reshape(-1, 1, 2)

    accepted = [point.copy() for point in existing if np.isfinite(point).all()]
    min_distance_squared = float(min_distance) ** 2
    bucket_size = max(float(min_distance), 1e-6)
    buckets: Dict[Tuple[int, int], list] = {}

    def key(point: np.ndarray) -> Tuple[int, int]:
        return (int(np.floor(float(point[0]) / bucket_size)), int(np.floor(float(point[1]) / bucket_size)))

    for index, point in enumerate(accepted):
        buckets.setdefault(key(point), []).append(index)

    for point in candidates:
        if not np.isfinite(point).all():
            continue
        bucket_x, bucket_y = key(point)
        too_close = False
        for delta_x in (-1, 0, 1):
            if too_close:
                break
            for delta_y in (-1, 0, 1):
                for other_index in buckets.get((bucket_x + delta_x, bucket_y + delta_y), ()):
                    difference = accepted[other_index] - point
                    if float(difference @ difference) < min_distance_squared:
                        too_close = True
                        break
                if too_close:
                    break
        if too_close:
            continue
        buckets.setdefault((bucket_x, bucket_y), []).append(len(accepted))
        accepted.append(point.copy())

    return np.asarray(accepted, dtype=np.float32).reshape(-1, 1, 2)


def replenish_points_grid(gray: np.ndarray, existing_points: np.ndarray, config: Mapping[str, object]) -> np.ndarray:
    detected = detect_points_grid(gray, config)
    return _merge_spaced_points(existing_points, detected, min_distance=float(config["min_distance"]))


def _lk_track(previous_gray, current_gray, points, config, min_valid):
    p0 = np.asarray(points, dtype=np.float32)
    if p0.ndim == 2 and p0.shape[1:] == (2,):
        p0 = p0.reshape(-1, 1, 2)
    if p0.ndim != 3 or p0.shape[1:] != (1, 2):
        raise ValueError("points must have shape N×2 or N×1×2")

    finite_input = np.isfinite(p0.reshape(-1, 2)).all(axis=1)
    p0 = p0[finite_input]
    if len(p0) < min_valid:
        return None

    lk_parameters = {
        "winSize": (int(config["lk_win_size"]), int(config["lk_win_size"])),
        "maxLevel": int(config["lk_max_level"]),
        "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        "minEigThreshold": float(config["min_eig_threshold"]),
    }

    p1, forward_status, forward_error = cv2.calcOpticalFlowPyrLK(previous_gray, current_gray, p0, None, **lk_parameters)
    if p1 is None or forward_status is None:
        return None

    p1 = np.asarray(p1, dtype=np.float32).reshape(-1, 1, 2)
    forward_status = np.asarray(forward_status, dtype=np.uint8).reshape(-1)
    if forward_error is None:
        forward_error = np.full(len(p0), np.nan, dtype=np.float32)
    else:
        forward_error = np.asarray(forward_error, dtype=np.float32).reshape(-1)

    p1_flat = p1.reshape(-1, 2)
    height, width = current_gray.shape
    p1_inside = (
        (p1_flat[:, 0] >= 0) & (p1_flat[:, 0] < width)
        & (p1_flat[:, 1] >= 0) & (p1_flat[:, 1] < height)
    )
    forward_eligible = (forward_status == 1) & np.isfinite(p1_flat).all(axis=1) & p1_inside
    eligible_indices = np.flatnonzero(forward_eligible)
    if len(eligible_indices) < min_valid:
        return None

    p0_back = np.full_like(p0, np.nan)
    backward_status = np.zeros(len(p0), dtype=np.uint8)
    p0_back_subset, backward_status_subset, _ = cv2.calcOpticalFlowPyrLK(
        current_gray, previous_gray, p1[eligible_indices], None, **lk_parameters,
    )
    if p0_back_subset is None or backward_status_subset is None:
        return None

    p0_back[eligible_indices] = np.asarray(p0_back_subset, dtype=np.float32).reshape(-1, 1, 2)
    backward_status[eligible_indices] = np.asarray(backward_status_subset, dtype=np.uint8).reshape(-1)

    p0_flat = p0.reshape(-1, 2)
    p0_back_flat = p0_back.reshape(-1, 2)
    finite_coordinates = (
        np.isfinite(p0_flat).all(axis=1)
        & np.isfinite(p1_flat).all(axis=1)
        & np.isfinite(p0_back_flat).all(axis=1)
    )
    fb_error = np.full(len(p0), np.inf, dtype=np.float32)
    comparable = forward_eligible & (backward_status == 1) & finite_coordinates
    fb_error[comparable] = np.linalg.norm(p0_flat[comparable] - p0_back_flat[comparable], axis=1)
    valid_mask = comparable & (fb_error <= float(config["fb_threshold"]))

    valid_count = int(np.count_nonzero(valid_mask))
    if valid_count < min_valid:
        return None

    return {
        "p0": p0, "p1": p1, "p0_back": p0_back,
        "forward_error": forward_error, "fb_error": fb_error,
        "valid_mask": valid_mask,
        "valid_p0": p0[valid_mask], "valid_p1": p1[valid_mask],
        "valid_count": valid_count,
    }


def track_points_forward_backward_relaxed(previous_gray, current_gray, points, config) -> Optional[dict]:
    if previous_gray.shape != current_gray.shape:
        raise ValueError("previous_gray and current_gray must have identical shapes")
    if previous_gray.ndim != 2 or previous_gray.dtype != np.uint8:
        raise ValueError("frames must be uint8 grayscale images")
    return _lk_track(previous_gray, current_gray, points, config, min_valid=4)


def track_points_forward_backward(previous_gray, current_gray, points, config) -> Optional[dict]:
    if previous_gray.shape != current_gray.shape:
        raise ValueError("previous_gray and current_gray must have identical shapes")
    if previous_gray.ndim != 2 or previous_gray.dtype != np.uint8:
        raise ValueError("frames must be uint8 grayscale images")
    return _lk_track(previous_gray, current_gray, points, config, min_valid=int(config["min_correspondences"]))


def _invalid_motion_result(motion_model, failure_reason, source_points=None, target_points=None,
                            matrix=None, inlier_mask=None, inlier_count=0, inlier_ratio=0.0,
                            model_parameters=None, model_confidence="invalid") -> dict:
    source_points = (
        np.asarray(source_points, dtype=np.float32).reshape(-1, 2)
        if source_points is not None else np.empty((0, 2), dtype=np.float32)
    )
    target_points = (
        np.asarray(target_points, dtype=np.float32).reshape(-1, 2)
        if target_points is not None else np.empty((0, 2), dtype=np.float32)
    )
    if inlier_mask is None:
        inlier_mask = np.zeros(len(source_points), dtype=bool)
    else:
        inlier_mask = np.asarray(inlier_mask, dtype=bool).reshape(-1)

    parameters = dict(model_parameters or {})
    return {
        "valid": False, "motion_model": str(motion_model),
        "matrix": None if matrix is None else np.asarray(matrix, dtype=np.float32),
        "inlier_mask": inlier_mask, "inlier_count": int(inlier_count), "inlier_ratio": float(inlier_ratio),
        "model_dx": float(parameters.get("dx", np.nan)), "model_dy": float(parameters.get("dy", np.nan)),
        "model_rotation_deg": float(parameters.get("rotation_deg", np.nan)),
        "model_scale": float(parameters.get("scale", np.nan)),
        "model_confidence": str(model_confidence), "failure_reason": str(failure_reason),
        "source_points": source_points, "target_points": target_points,
    }


def _motion_model_parameters(matrix: np.ndarray) -> dict:
    matrix = np.asarray(matrix, dtype=np.float64)

    if matrix.shape == (2, 3):
        normalized = matrix
    elif matrix.shape == (3, 3):
        if abs(matrix[2, 2]) < 1e-12:
            raise ValueError("homography has near-zero H[2,2]")
        normalized = matrix / matrix[2, 2]
    else:
        raise ValueError("motion matrix must have shape 2×3 or 3×3")

    linear = normalized[:2, :2]
    determinant = float(np.linalg.det(linear))
    scale = float(np.sqrt(abs(determinant)))
    rotation_deg = float(np.degrees(np.arctan2(linear[1, 0], linear[0, 0])))
    dx = float(normalized[0, 2])
    dy = float(normalized[1, 2])

    return {"dx": dx, "dy": dy, "rotation_deg": rotation_deg, "scale": scale, "determinant": determinant}


def estimate_camera_motion(p0: np.ndarray, p1: np.ndarray, config: Mapping[str, object], frame_shape: Tuple[int, ...]) -> dict:
    source = np.asarray(p0, dtype=np.float32).reshape(-1, 2)
    target = np.asarray(p1, dtype=np.float32).reshape(-1, 2)
    motion_model = str(config["motion_model"])
    min_correspondences = int(config["min_correspondences"])

    if source.shape != target.shape:
        raise ValueError("p0 and p1 must have identical shapes")

    finite = np.isfinite(source).all(axis=1) & np.isfinite(target).all(axis=1)
    source = source[finite]
    target = target[finite]

    if len(source) < min_correspondences:
        return _invalid_motion_result(
            motion_model, f"not enough finite correspondences: {len(source)} < {min_correspondences}", source, target,
        )

    ransac_threshold = float(config["ransac_threshold"])
    if motion_model == "affine_partial":
        matrix, inlier_mask = cv2.estimateAffinePartial2D(
            source, target, method=cv2.RANSAC, ransacReprojThreshold=ransac_threshold,
            maxIters=2000, confidence=0.99, refineIters=10,
        )
    elif motion_model == "affine_full":
        matrix, inlier_mask = cv2.estimateAffine2D(
            source, target, method=cv2.RANSAC, ransacReprojThreshold=ransac_threshold,
            maxIters=2000, confidence=0.99, refineIters=10,
        )
    elif motion_model == "homography":
        matrix, inlier_mask = cv2.findHomography(
            source, target, method=cv2.RANSAC, ransacReprojThreshold=ransac_threshold,
            maxIters=2000, confidence=0.99,
        )
    else:
        raise ValueError("motion_model must be affine_partial, affine_full or homography")

    if matrix is None or inlier_mask is None:
        return _invalid_motion_result(motion_model, f"{motion_model} RANSAC could not estimate a model", source, target)

    matrix = np.asarray(matrix, dtype=np.float64)
    expected_shapes = {"affine_partial": (2, 3), "affine_full": (2, 3), "homography": (3, 3)}
    if matrix.shape != expected_shapes[motion_model]:
        return _invalid_motion_result(motion_model, f"invalid matrix shape: {matrix.shape}", source, target, matrix=matrix)
    if not np.isfinite(matrix).all():
        return _invalid_motion_result(motion_model, "model matrix contains NaN or Inf", source, target, matrix=matrix)

    if motion_model == "homography":
        if abs(matrix[2, 2]) < 1e-12:
            return _invalid_motion_result(
                motion_model, "degenerate: homography H[2,2] is near zero", source, target, matrix=matrix,
            )
        matrix = matrix / matrix[2, 2]

    inlier_mask = np.asarray(inlier_mask, dtype=bool).reshape(-1)
    if len(inlier_mask) != len(source):
        return _invalid_motion_result(
            motion_model, "RANSAC inlier mask has the wrong length", source, target, matrix=matrix,
        )

    inlier_count = int(np.count_nonzero(inlier_mask))
    inlier_ratio = float(np.mean(inlier_mask))

    try:
        parameters = _motion_model_parameters(matrix)
    except Exception as error:
        return _invalid_motion_result(
            motion_model, f"degenerate: {error}", source, target, matrix=matrix,
            inlier_mask=inlier_mask, inlier_count=inlier_count, inlier_ratio=inlier_ratio,
        )

    preliminary_confidence = "low" if inlier_ratio < float(config["low_confidence_inlier_ratio"]) else "high"

    def invalid_with_diagnostics(reason: str) -> dict:
        return _invalid_motion_result(
            motion_model, reason, source, target, matrix=matrix, inlier_mask=inlier_mask,
            inlier_count=inlier_count, inlier_ratio=inlier_ratio,
            model_parameters=parameters, model_confidence=preliminary_confidence,
        )

    frame_height = int(frame_shape[0])
    frame_width = int(frame_shape[1])

    if inlier_count < int(config["min_ransac_inliers"]):
        return invalid_with_diagnostics(f"not enough RANSAC inliers: {inlier_count} < {config['min_ransac_inliers']}")
    if parameters["determinant"] <= 0:
        return invalid_with_diagnostics("degenerate: reflection")
    if not (float(config["min_model_scale"]) <= parameters["scale"] <= float(config["max_model_scale"])):
        return invalid_with_diagnostics(f"degenerate: scale={parameters['scale']:.2f}")
    if abs(parameters["dx"]) > 0.5 * frame_width:
        return invalid_with_diagnostics("degenerate: dx too large")
    if abs(parameters["dy"]) > 0.5 * frame_height:
        return invalid_with_diagnostics("degenerate: dy too large")
    if inlier_ratio < float(config["min_inlier_ratio"]):
        return invalid_with_diagnostics(f"low inlier ratio {inlier_ratio:.2f}")

    model_confidence = "low" if inlier_ratio < float(config["low_confidence_inlier_ratio"]) else "high"
    return {
        "valid": True, "motion_model": motion_model, "matrix": matrix.astype(np.float32),
        "inlier_mask": inlier_mask, "inlier_count": inlier_count, "inlier_ratio": inlier_ratio,
        "model_dx": parameters["dx"], "model_dy": parameters["dy"],
        "model_rotation_deg": parameters["rotation_deg"], "model_scale": parameters["scale"],
        "model_confidence": model_confidence, "failure_reason": None,
        "source_points": source, "target_points": target,
    }


def _warped_validity_polygon(image_shape: Tuple[int, int], motion_matrix: np.ndarray) -> np.ndarray:
    height, width = map(int, image_shape)
    corners = np.asarray(
        [[0.0, 0.0], [float(width - 1), 0.0], [float(width - 1), float(height - 1)], [0.0, float(height - 1)]],
        dtype=np.float32,
    )

    matrix = np.asarray(motion_matrix, dtype=np.float32)
    if matrix.shape == (2, 3):
        transformed = cv2.transform(corners.reshape(1, -1, 2), matrix)[0]
    elif matrix.shape == (3, 3):
        transformed = cv2.perspectiveTransform(corners.reshape(1, -1, 2), matrix)[0]
    else:
        raise ValueError("motion_matrix must have shape 2×3 or 3×3")

    if not np.isfinite(transformed).all():
        raise RuntimeError("Warped validity polygon contains NaN or Inf")

    polygon = np.rint(transformed).astype(np.int32)
    validity_u8 = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(validity_u8, polygon, 255, lineType=cv2.LINE_8)
    return validity_u8 > 127


def warp_grayscale_fast(previous_gray: np.ndarray, motion_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if not isinstance(previous_gray, np.ndarray):
        raise ValueError("previous_gray must be a numpy array")
    if previous_gray.ndim != 2 or previous_gray.dtype != np.uint8:
        raise ValueError("previous_gray must be a uint8 grayscale image")
    if previous_gray.size == 0:
        raise ValueError("previous_gray must not be empty")

    motion_matrix = np.asarray(motion_matrix, dtype=np.float32)
    if motion_matrix.shape not in {(2, 3), (3, 3)}:
        raise ValueError("motion_matrix must have shape 2×3 or 3×3")
    if not np.isfinite(motion_matrix).all():
        raise ValueError("motion_matrix must contain only finite values")

    height, width = previous_gray.shape
    output_size = (width, height)

    if motion_matrix.shape == (2, 3):
        warped_previous = cv2.warpAffine(
            previous_gray, motion_matrix, output_size,
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )
    else:
        warped_previous = cv2.warpPerspective(
            previous_gray, motion_matrix, output_size,
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )

    validity_mask = _warped_validity_polygon(previous_gray.shape, motion_matrix)
    if not np.any(validity_mask):
        raise RuntimeError("Warping produced an empty validity mask")

    return warped_previous, validity_mask


def _as_homography(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape == (2, 3):
        homography = np.eye(3, dtype=np.float64)
        homography[:2, :] = matrix
    elif matrix.shape == (3, 3):
        homography = matrix.copy()
    else:
        raise ValueError("motion matrix must have shape 2×3 or 3×3")

    if not np.isfinite(homography).all():
        raise ValueError("motion matrix contains NaN or Inf")
    if abs(homography[2, 2]) < 1e-12:
        raise ValueError("homography H[2,2] is near zero")
    return (homography / homography[2, 2]).astype(np.float32)


def _natural_sort_key(path: Path) -> tuple:
    return tuple(
        int(part) if part.isdigit() else part.casefold()
        for part in re.split(r"(\d+)", path.name)
    )


class CompensationPipeline:
    """Stateful compensation pipeline (production, step=1)."""

    def __init__(self, config: Mapping[str, object] = PRODUCTION_CONFIG, step: int = 1) -> None:
        if int(step) <= 0:
            raise ValueError("step must be a positive integer")

        self.config = _validate_config(config)
        self.step = int(step)
        self.frames: Deque[Tuple[int, np.ndarray]] = deque(maxlen=2 * self.step + 1)
        self.matrices: Dict[int, np.ndarray] = {}
        self.stats = {}
        self.last_failure_reason: Optional[str] = None
        self._motion_cache: Dict[int, _MotionCacheEntry] = {}
        self._next_frame_index = 0
        self._motion_estimation_calls = 0
        self._processing_scale: Optional[float] = None
        self._frame_shape: Optional[Tuple[int, int]] = None

        self._tracked_points = np.empty((0, 1, 2), dtype=np.float32)
        self._frames_since_redetect = 0
        self._point_detection_calls = 0
        self._periodic_redetections = 0
        self._threshold_replenishments = 0
        self._tensor_buffer: Optional[np.ndarray] = None
        self._decoder_target_size: Optional[Tuple[int, int]] = None
        self._reset_stats()

    def _reset_stats(self) -> None:
        self.stats = {
            "emitted": 0, "warmup": 0, "invalid_compensation": 0,
            "point_detection_calls": 0, "periodic_redetections": 0,
            "threshold_replenishments": 0, "decoder_scaled_frames": 0,
        }

    @property
    def motion_estimation_calls(self) -> int:
        return self._motion_estimation_calls

    @property
    def point_detection_calls(self) -> int:
        return self._point_detection_calls

    def reset(self) -> None:
        self.frames.clear()
        self.matrices.clear()
        self._motion_cache.clear()
        self._reset_stats()
        self.last_failure_reason = None
        self._next_frame_index = 0
        self._motion_estimation_calls = 0
        self._processing_scale = None
        self._frame_shape = None
        self._tracked_points = np.empty((0, 1, 2), dtype=np.float32)
        self._frames_since_redetect = 0
        self._point_detection_calls = 0
        self._periodic_redetections = 0
        self._threshold_replenishments = 0
        self._tensor_buffer = None
        self._decoder_target_size = None

    def _frame_at(self, frame_index: int) -> np.ndarray:
        for cached_index, gray in self.frames:
            if cached_index == frame_index:
                return gray
        raise KeyError(f"frame {frame_index} is not present in the rolling buffer")

    def _detect_points(self, gray: np.ndarray) -> np.ndarray:
        self._point_detection_calls += 1
        self.stats["point_detection_calls"] = self._point_detection_calls
        return detect_points_grid(gray, self.config)

    def _replenish_points(self, gray: np.ndarray, existing_points: np.ndarray) -> np.ndarray:
        self._point_detection_calls += 1
        self.stats["point_detection_calls"] = self._point_detection_calls
        return replenish_points_grid(gray, existing_points, self.config)

    def _select_current_tracks(self, tracking: Optional[dict], motion: Optional[dict]) -> np.ndarray:
        if tracking is None:
            return np.empty((0, 1, 2), dtype=np.float32)

        current_points = np.asarray(tracking["valid_p1"], dtype=np.float32).reshape(-1, 1, 2)

        if motion is None:
            return current_points

        inlier_mask = np.asarray(motion.get("inlier_mask", []), dtype=bool).reshape(-1)
        if len(inlier_mask) == len(current_points) and np.any(inlier_mask):
            return current_points[inlier_mask]
        return current_points

    def _refresh_tracks_for_next_frame(self, current_gray: np.ndarray, current_tracks: np.ndarray) -> bool:
        self._frames_since_redetect += 1
        redetect_interval = int(self.config["redetect_interval"])
        valid_count = int(len(current_tracks))

        if self._frames_since_redetect >= redetect_interval:
            self._tracked_points = self._detect_points(current_gray)
            self._frames_since_redetect = 0
            self._periodic_redetections += 1
            self.stats["periodic_redetections"] = self._periodic_redetections
            return True

        if valid_count < int(self.config["min_correspondences"]):
            self._tracked_points = self._replenish_points(current_gray, current_tracks)
            self._frames_since_redetect = 0
            self._threshold_replenishments += 1
            self.stats["threshold_replenishments"] = self._threshold_replenishments
            return True

        self._tracked_points = current_tracks
        return False

    def _estimate_pair_motion(self, previous_gray: np.ndarray, current_gray: np.ndarray) -> _MotionCacheEntry:
        self._motion_estimation_calls += 1

        if previous_gray.shape != current_gray.shape:
            self._tracked_points = self._detect_points(current_gray)
            self._frames_since_redetect = 0
            return _MotionCacheEntry(
                valid=False, matrix=None, inlier_ratio=0.0,
                failure_reason="frame sizes do not match after preprocessing",
            )

        try:
            if len(self._tracked_points) < 4:
                self._tracked_points = self._detect_points(previous_gray)
                self._frames_since_redetect = 0

            tracking = track_points_forward_backward_relaxed(
                previous_gray, current_gray, self._tracked_points, self.config,
            )

            valid_count = int(tracking["valid_count"]) if tracking is not None else 0
            motion = None
            if tracking is not None and valid_count >= int(self.config["min_correspondences"]):
                motion = estimate_camera_motion(
                    tracking["valid_p0"], tracking["valid_p1"], self.config, current_gray.shape,
                )

            current_tracks = self._select_current_tracks(tracking, motion)
            self._refresh_tracks_for_next_frame(current_gray, current_tracks)

            if tracking is None:
                return _MotionCacheEntry(
                    valid=False, matrix=None, inlier_ratio=0.0,
                    failure_reason="forward/backward LK produced fewer than 4 tracks",
                )

            if valid_count < int(self.config["min_correspondences"]):
                return _MotionCacheEntry(
                    valid=False, matrix=None, inlier_ratio=0.0,
                    failure_reason=(
                        f"valid LK correspondences dropped to {valid_count} < "
                        f"{self.config['min_correspondences']}; points replenished"
                    ),
                )

            assert motion is not None
            if not motion["valid"]:
                return _MotionCacheEntry(
                    valid=False, matrix=None, inlier_ratio=float(motion["inlier_ratio"]),
                    failure_reason=str(motion["failure_reason"]),
                )

            return _MotionCacheEntry(
                valid=True, matrix=_as_homography(motion["matrix"]),
                inlier_ratio=float(motion["inlier_ratio"]), failure_reason=None,
            )
        except (cv2.error, RuntimeError, ValueError) as error:
            self._tracked_points = self._detect_points(current_gray)
            self._frames_since_redetect = 0
            return _MotionCacheEntry(
                valid=False, matrix=None, inlier_ratio=0.0, failure_reason=f"compensation failed: {error}",
            )

    def _prune_cache(self, current_index: int) -> None:
        oldest_allowed = current_index - 2 * self.step
        for key in list(self._motion_cache):
            if key < oldest_allowed:
                del self._motion_cache[key]
                self.matrices.pop(key, None)

    def _invalid_sample(self, frame_index, skip_reason, inlier_ratio=0.0, failure_reason=None) -> CompensatedSample:
        self.stats[skip_reason] += 1
        self.last_failure_reason = failure_reason
        return CompensatedSample(
            tensor=_EMPTY_TENSOR.copy(), frame_index=frame_index, valid=False,
            inlier_ratio=float(inlier_ratio), skip_reason=skip_reason,
        )

    def _build_tensor(self, warped_oldest, warped_middle, current_gray, validity) -> np.ndarray:
        height, width = current_gray.shape
        shape = (4, height, width)
        if self._tensor_buffer is None or self._tensor_buffer.shape != shape:
            self._tensor_buffer = np.empty(shape, dtype=np.float32)

        scale = np.float32(1.0 / 255.0)
        np.multiply(warped_oldest, scale, out=self._tensor_buffer[0], casting="unsafe")
        np.multiply(warped_middle, scale, out=self._tensor_buffer[1], casting="unsafe")
        np.multiply(current_gray, scale, out=self._tensor_buffer[2], casting="unsafe")
        np.copyto(self._tensor_buffer[3], validity, casting="unsafe")
        return self._tensor_buffer

    def process_frame(self, frame: np.ndarray) -> CompensatedSample:
        """Обробляє один BGR кадр і повертає один семпл (класична частина)."""
        frame_index = self._next_frame_index
        self._next_frame_index += 1

        gray, scale = preprocess_frame(
            frame, int(self.config["max_width"]), gray_first=bool(self.config["preprocess_gray_first"]),
        )
        if self._frame_shape is None:
            self._frame_shape = gray.shape
            self._processing_scale = scale
        self.frames.append((frame_index, gray))

        if frame_index == 0:
            self._tracked_points = self._detect_points(gray)
            self._frames_since_redetect = 0

        current_motion: Optional[_MotionCacheEntry] = None
        if frame_index >= self.step:
            previous_gray = self._frame_at(frame_index - self.step)
            current_motion = self._estimate_pair_motion(previous_gray, gray)
            self._motion_cache[frame_index] = current_motion
            if current_motion.valid and current_motion.matrix is not None:
                self.matrices[frame_index] = current_motion.matrix

        self._prune_cache(frame_index)

        if frame_index < 2 * self.step:
            return self._invalid_sample(frame_index=frame_index, skip_reason="warmup", failure_reason=None)

        previous_target_index = frame_index - self.step
        previous_motion = self._motion_cache.get(previous_target_index)
        if current_motion is None:
            current_motion = self._motion_cache.get(frame_index)

        entries = (previous_motion, current_motion)
        available_ratios = [entry.inlier_ratio for entry in entries if entry is not None]
        weakest_ratio = min(available_ratios, default=0.0)

        if previous_motion is None or current_motion is None:
            return self._invalid_sample(
                frame_index=frame_index, skip_reason="invalid_compensation",
                inlier_ratio=weakest_ratio, failure_reason="required cached motion is missing",
            )

        invalid_entry = next(
            (entry for entry in entries if not entry.valid or entry.matrix is None), None,
        )
        if invalid_entry is not None:
            return self._invalid_sample(
                frame_index=frame_index, skip_reason="invalid_compensation",
                inlier_ratio=weakest_ratio, failure_reason=invalid_entry.failure_reason,
            )

        assert previous_motion.matrix is not None
        assert current_motion.matrix is not None

        try:
            composed_matrix = current_motion.matrix @ previous_motion.matrix
            composed_matrix = _as_homography(composed_matrix)

            oldest_gray = self._frame_at(frame_index - 2 * self.step)
            middle_gray = self._frame_at(frame_index - self.step)

            warped_oldest, validity_oldest = warp_grayscale_fast(oldest_gray, composed_matrix)
            warped_middle, validity_middle = warp_grayscale_fast(middle_gray, current_motion.matrix)
            validity = np.logical_and(validity_oldest, validity_middle)
            if not np.any(validity):
                raise RuntimeError("intersection of validity masks is empty")

            tensor = self._build_tensor(warped_oldest, warped_middle, gray, validity)
        except (cv2.error, RuntimeError, ValueError) as error:
            return self._invalid_sample(
                frame_index=frame_index, skip_reason="invalid_compensation",
                inlier_ratio=weakest_ratio, failure_reason=f"tensor construction failed: {error}",
            )

        self.stats["emitted"] += 1
        self.last_failure_reason = None
        return CompensatedSample(
            tensor=tensor, frame_index=frame_index, valid=True,
            inlier_ratio=float(weakest_ratio), skip_reason=None,
        )

    def stream_video(self, video_path) -> Iterator[CompensatedSample]:
        path = Path(video_path)
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise FileNotFoundError(f"Could not open video: {path}")

        source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.reset()

        if bool(self.config["request_decoder_scaling"]):
            self._decoder_target_size = _request_decoder_scaling(capture, int(self.config["max_width"]))

        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break

                if (
                    self._decoder_target_size is not None
                    and source_width > int(self.config["max_width"])
                    and tuple(frame.shape[1::-1]) == tuple(self._decoder_target_size)
                ):
                    self.stats["decoder_scaled_frames"] += 1

                yield self.process_frame(frame)
        finally:
            capture.release()


# ================================================================
# 2. TinyUNet (нейромережа) + inference-only Conv+BN fusion
# ================================================================

class DoubleConv(nn.Module):
    """Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> ReLU"""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class TinyUNet(nn.Module):
    """
    Вхід:  (B, 3, H, W) — три вирівняних grayscale кадри як канали.
    Вихід: (B, 1, H, W) — LOGITS (sigmoid застосовується окремо при inference).
    """

    def __init__(self):
        super().__init__()
        self.enc1 = DoubleConv(3, 16)
        self.enc2 = DoubleConv(16, 32)
        self.enc3 = DoubleConv(32, 48)
        self.pool = nn.MaxPool2d(2)
        self.bott = DoubleConv(48, 64)
        self.dec3 = DoubleConv(64 + 48, 48)
        self.dec2 = DoubleConv(48 + 32, 32)
        self.dec1 = DoubleConv(32 + 16, 16)
        self.head = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bott(self.pool(e3))

        u3 = F.interpolate(b, size=e3.shape[2:], mode="bilinear", align_corners=False)
        d3 = self.dec3(torch.cat([u3, e3], dim=1))

        u2 = F.interpolate(d3, size=e2.shape[2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))

        u1 = F.interpolate(d2, size=e1.shape[2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))

        return self.head(d1)


def fuse_tiny_unet_batchnorm(model, inplace=False):
    """Об'єднує кожну пару Conv2d+BatchNorm2d в TinyUNet в один Conv2d (лише inference)."""
    fused_model = model if inplace else copy.deepcopy(model)
    fused_model.eval()

    fused_pairs = 0
    for module in fused_model.modules():
        if not isinstance(module, DoubleConv):
            continue

        block = module.block
        for conv_index, bn_index in ((0, 1), (3, 4)):
            conv = block[conv_index]
            bn = block[bn_index]

            if isinstance(conv, nn.Conv2d) and isinstance(bn, nn.BatchNorm2d):
                block[conv_index] = torch.nn.utils.fusion.fuse_conv_bn_eval(conv, bn)
                block[bn_index] = nn.Identity()
                fused_pairs += 1

    remaining_bn = sum(isinstance(module, nn.BatchNorm2d) for module in fused_model.modules())
    if remaining_bn != 0:
        raise RuntimeError(f"BatchNorm fusion incomplete: {remaining_bn} BN layers remain")

    fused_model._fused_batchnorm_pairs = int(fused_pairs)
    return fused_model


def build_inference_model(state_dict, device=DEVICE, fuse_batchnorm=FUSE_BATCHNORM_FOR_INFERENCE):
    """Будує eval-only TinyUNet, зливає BN на CPU, переносить на device."""
    inference_model = TinyUNet()
    inference_model.load_state_dict(state_dict)
    inference_model.eval()

    if fuse_batchnorm:
        inference_model = fuse_tiny_unet_batchnorm(inference_model, inplace=True)
        print("Fused Conv+BN pairs:", inference_model._fused_batchnorm_pairs)

    return inference_model.to(device).eval()


def load_frozen_pytorch_model(checkpoint_path, device=DEVICE) -> TinyUNet:
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint)
    return build_inference_model(state_dict, device=device, fuse_batchnorm=True)


# ================================================================
# 2b. PROBABILITY BACKENDS (PyTorch / Core ML) + експорт під Apple
# ================================================================

def _sync_device(device: str) -> None:
    """Форсує завершення асинхронних GPU-операцій.

    Без цього таймер міряє час ПОСТАНОВКИ в чергу, а не обчислення,
    і стадія 'inference' покаже фантастичні 0.3 мс.
    """
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        mps_module = getattr(torch, "mps", None)
        if mps_module is not None and hasattr(mps_module, "synchronize"):
            mps_module.synchronize()


class ProbabilityBackend:
    """
    Спільний інтерфейс для нейронної частини.

    Реалізації взаємозамінні в `predict_video_bboxes_pipelined`, тому
    перехід PyTorch -> Core ML не зачіпає ні конвеєр, ні постобробку.

    `infer` повертає бінарну uint8 маску HxW (0/255) — саме її очікує
    `_probability_to_blobs`.
    `infer_profiled` додатково повертає розбивку по стадіях.
    """

    name = "base"

    def infer(self, input_chw: np.ndarray, threshold: float) -> np.ndarray:
        raise NotImplementedError

    def infer_profiled(self, input_chw: np.ndarray, threshold: float) -> Tuple[np.ndarray, dict]:
        raise NotImplementedError

    def probability(self, input_chw: np.ndarray) -> np.ndarray:
        """Float-мапа ймовірностей. Потрібна лише для діагностики/калібрування."""
        raise NotImplementedError

    def warmup(self, input_chw: np.ndarray, threshold: float,
               iterations: int = NEURAL_WARMUP_ITERATIONS) -> dict:
        started = time.perf_counter()
        for _ in range(max(0, int(iterations))):
            self.infer(input_chw, threshold)
        return {
            "backend": self.name,
            "iterations": int(iterations),
            "total_ms": (time.perf_counter() - started) * 1000.0,
        }

    def close(self) -> None:
        return None


class TorchProbabilityBackend(ProbabilityBackend):
    """PyTorch-бекенд (CUDA / MPS / CPU). Поріг застосовується на девайсі."""

    name = "torch"

    def __init__(self, model: nn.Module, device: str = DEVICE):
        self.model = model.eval()
        self.device = str(device)
        self.name = f"torch:{self.device}"

    def _to_device(self, input_chw: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(input_chw).unsqueeze(0).to(
            self.device, non_blocking=(self.device == "cuda"),
        )

    def infer(self, input_chw: np.ndarray, threshold: float) -> np.ndarray:
        with torch.inference_mode():
            x = self._to_device(input_chw)
            logits = self.model(x)
            probability = torch.sigmoid(logits)[0, 0]
            # Поріг НА ДЕВАЙСІ: назад іде uint8 (230 KB), а не float32 (920 KB).
            binary = (probability > float(threshold)).to(torch.uint8).mul_(255)
            return binary.cpu().numpy()

    def infer_profiled(self, input_chw: np.ndarray, threshold: float) -> Tuple[np.ndarray, dict]:
        with torch.inference_mode():
            _sync_device(self.device)
            t0 = time.perf_counter()
            x = self._to_device(input_chw)
            _sync_device(self.device)
            t1 = time.perf_counter()

            logits = self.model(x)
            probability = torch.sigmoid(logits)[0, 0]
            binary = (probability > float(threshold)).to(torch.uint8).mul_(255)
            _sync_device(self.device)
            t2 = time.perf_counter()

            mask = binary.cpu().numpy()
            t3 = time.perf_counter()

        timings = {
            "h2d_ms": (t1 - t0) * 1000.0,
            "compute_ms": (t2 - t1) * 1000.0,
            "d2h_ms": (t3 - t2) * 1000.0,
        }
        timings["total_ms"] = sum(timings.values())
        return mask, timings

    def probability(self, input_chw: np.ndarray) -> np.ndarray:
        with torch.inference_mode():
            x = self._to_device(input_chw)
            return torch.sigmoid(self.model(x))[0, 0].detach().cpu().numpy()


class CoreMLProbabilityBackend(ProbabilityBackend):
    """
    Core ML бекенд (fp16, ANE/GPU) — цільовий для ARM Mac.

    Модель віддає float-мапу ймовірностей, поріг застосовується в numpy.
    Так зроблено свідомо: поріг ще калібрується, і запікати його в граф
    означало б переекспортовувати модель на кожне значення. Порівняння
    230k float'ів коштує ~0.1 мс і не є вузьким місцем.
    """

    name = "coreml"

    def __init__(self, model_path, compute_units: str = "ALL"):
        try:
            import coremltools as ct
        except ImportError as error:
            raise ImportError(
                "coremltools не встановлено. Core ML бекенд доступний лише на macOS: "
                "pip install coremltools"
            ) from error

        units = getattr(ct.ComputeUnit, str(compute_units).upper(), ct.ComputeUnit.ALL)
        self.model = ct.models.MLModel(str(model_path), compute_units=units)
        self.compute_units = str(compute_units).upper()
        self.name = f"coreml:{self.compute_units}"

        spec = self.model.get_spec()
        self.input_name = str(spec.description.input[0].name)
        self.output_name = str(spec.description.output[0].name)

    def _raw_probability(self, input_chw: np.ndarray) -> np.ndarray:
        batch = np.ascontiguousarray(input_chw, dtype=np.float32)[None, ...]
        prediction = self.model.predict({self.input_name: batch})
        return np.asarray(prediction[self.output_name], dtype=np.float32).reshape(
            batch.shape[2], batch.shape[3],
        )

    def infer(self, input_chw: np.ndarray, threshold: float) -> np.ndarray:
        probability = self._raw_probability(input_chw)
        return (probability > float(threshold)).astype(np.uint8) * 255

    def infer_profiled(self, input_chw: np.ndarray, threshold: float) -> Tuple[np.ndarray, dict]:
        t0 = time.perf_counter()
        probability = self._raw_probability(input_chw)
        t1 = time.perf_counter()
        mask = (probability > float(threshold)).astype(np.uint8) * 255
        t2 = time.perf_counter()

        # Python-API Core ML не розділяє перенос буфера і виконання графа,
        # тому h2d/compute/d2h чесно склеєні в один непрозорий блок.
        # Вигадувати правдоподібну розбивку тут було б брехнею.
        return mask, {
            "h2d_ms": float("nan"),
            "compute_ms": (t1 - t0) * 1000.0,
            "d2h_ms": float("nan"),
            "threshold_ms": (t2 - t1) * 1000.0,
            "total_ms": (t2 - t0) * 1000.0,
            "opaque_transfer": True,
        }

    def probability(self, input_chw: np.ndarray) -> np.ndarray:
        return self._raw_probability(input_chw)


class _TinyUNetProbabilityWrapper(nn.Module):
    """Обгортка для експорту: фіксований вхід -> мапа ймовірностей."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model.eval()

    def forward(self, frames):
        return torch.sigmoid(self.model(frames))


def export_tinyunet_coreml_fp16(
    checkpoint_path, output_path, input_shape=(1, 3, 360, 640),
    compute_units: str = "ALL", traced_output_path=None,
) -> dict:
    """
    Експортує TinyUNet у Core ML ML Program (fp16) під Apple Silicon.

    Форма входу ФІКСОВАНА, а не RangeDim: роздільна здатність обробки
    константна (PROCESSING_MAX_WIDTH), а динамічні shape'и — найчастіша
    причина, через яку Core ML відмовляється класти граф на ANE.
    """
    try:
        import coremltools as ct
    except ImportError as error:
        raise ImportError(
            "coremltools не встановлено. Експорт можливий лише на macOS: pip install coremltools"
        ) from error

    height, width = int(input_shape[2]), int(input_shape[3])
    if height % 8 != 0 or width % 8 != 0:
        raise ValueError(f"TinyUNet вимагає H і W кратні 8, отримано {(height, width)}")

    # Експортуємо з CPU-копії з уже вплавленим BatchNorm.
    model = load_frozen_pytorch_model(checkpoint_path, device="cpu")
    wrapper = _TinyUNetProbabilityWrapper(model).eval()

    example = torch.zeros(tuple(int(v) for v in input_shape), dtype=torch.float32)
    with torch.inference_mode():
        traced = torch.jit.trace(wrapper, example)
    if traced_output_path is not None:
        Path(traced_output_path).parent.mkdir(parents=True, exist_ok=True)
        traced.save(str(traced_output_path))

    units = getattr(ct.ComputeUnit, str(compute_units).upper(), ct.ComputeUnit.ALL)
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="frames", shape=tuple(int(v) for v in input_shape))],
        outputs=[ct.TensorType(name="probability")],
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        compute_units=units,
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(output_path))

    return {
        "output_path": str(output_path),
        "input_shape": tuple(int(v) for v in input_shape),
        "precision": "float16",
        "compute_units": str(compute_units).upper(),
        "traced_path": str(traced_output_path) if traced_output_path else None,
    }


def verify_coreml_parity(
    torch_backend: TorchProbabilityBackend, coreml_backend: CoreMLProbabilityBackend,
    video_path, max_samples: int = 100, compensation_config=None,
) -> dict:
    """
    Звіряє fp16 Core ML з fp32 PyTorch на реальних кадрах.

    Синтетичний шум тут не годиться: розподіл активацій на справжніх
    compensated-тензорах інший, і саме він визначає, чи безпечний fp16.
    """
    config = dict(PRODUCTION_CONFIG)
    if compensation_config is not None:
        config.update(dict(compensation_config))
    config["max_width"] = PROCESSING_MAX_WIDTH

    pipeline = CompensationPipeline(config, step=1)
    abs_errors: List[float] = []
    max_error = 0.0
    samples = 0

    for sample in pipeline.stream_video(video_path):
        if not sample.valid:
            continue
        source = _prepare_unet_input(np.asarray(sample.tensor), checkpoint_blur=False, out=None)
        reference = torch_backend.probability(source)
        candidate = coreml_backend.probability(source)
        difference = np.abs(reference.astype(np.float64) - candidate.astype(np.float64))
        abs_errors.append(float(difference.mean()))
        max_error = max(max_error, float(difference.max()))
        samples += 1
        if samples >= int(max_samples):
            break

    if samples == 0:
        raise RuntimeError("Не вдалося отримати жодного валідного кадру для перевірки parity")

    return {
        "samples": samples,
        "mean_absolute_error": float(np.mean(abs_errors)),
        "max_absolute_error": max_error,
        "note": "MAE < 1e-3 — fp16 безпечний. Більше 1e-2 — перевіряти окремі шари.",
    }


def build_probability_backend(
    checkpoint_path=None, coreml_path=None, device: str = DEVICE,
    compute_units: str = "ALL", model: Optional[nn.Module] = None,
) -> ProbabilityBackend:
    """Обирає бекенд: Core ML, якщо вказано .mlpackage, інакше PyTorch."""
    if coreml_path is not None:
        return CoreMLProbabilityBackend(coreml_path, compute_units=compute_units)
    if model is None:
        if checkpoint_path is None:
            raise ValueError("Потрібен checkpoint_path, coreml_path або готовий model")
        model = load_frozen_pytorch_model(checkpoint_path, device=device)
    return TorchProbabilityBackend(model, device=device)


# ================================================================
# 3. SORT tracker
# ================================================================

def iou(bb_test, bb_gt):
    xx1 = max(bb_test[0], bb_gt[0]); yy1 = max(bb_test[1], bb_gt[1])
    xx2 = min(bb_test[2], bb_gt[2]); yy2 = min(bb_test[3], bb_gt[3])
    w = max(0., xx2 - xx1); h = max(0., yy2 - yy1); inter = w * h
    union = ((bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1]) +
             (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1]) - inter)
    return inter / union if union > 0 else 0.0


def center_distance(bb_test, bb_gt):
    cx1 = (bb_test[0] + bb_test[2]) / 2; cy1 = (bb_test[1] + bb_test[3]) / 2
    cx2 = (bb_gt[0] + bb_gt[2]) / 2; cy2 = (bb_gt[1] + bb_gt[3]) / 2
    return np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)


class Track:
    def __init__(self, bbox, track_id):
        self.track_id = track_id; self.hits = 1; self.hit_streak = 1
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kf.processNoiseCov = np.diag([1., 1., 10., 10.]).astype(np.float32)
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 5.0
        self.kf.errorCovPost = np.eye(4, dtype=np.float32) * 100.0
        cx = (bbox[0] + bbox[2]) / 2; cy = (bbox[1] + bbox[3]) / 2
        self.kf.statePre = np.array([[cx], [cy], [0], [0]], np.float32)
        self.kf.statePost = np.array([[cx], [cy], [0], [0]], np.float32)
        self.time_since_update = 0; self.history = deque(maxlen=60)
        self.bbox = list(bbox); self.w = bbox[2] - bbox[0]; self.h = bbox[3] - bbox[1]

    def predict(self):
        pred = self.kf.predict()
        cx, cy = float(pred[0, 0]), float(pred[1, 0])
        self.bbox = [cx - self.w / 2, cy - self.h / 2, cx + self.w / 2, cy + self.h / 2]
        self.time_since_update += 1; return self.bbox

    def update(self, bbox):
        self.time_since_update = 0; self.hits += 1; self.hit_streak += 1
        self.bbox = list(bbox); self.w = bbox[2] - bbox[0]; self.h = bbox[3] - bbox[1]
        cx = (bbox[0] + bbox[2]) / 2; cy = (bbox[1] + bbox[3]) / 2
        self.history.append((int(cx), int(cy)))
        self.kf.correct(np.array([[cx], [cy]], np.float32))

    def miss(self):
        self.hit_streak = 0


class SortTracker:
    def __init__(self, max_age=15, min_hits=2, iou_threshold=0.05, max_dist=80):
        self.max_age = max_age; self.min_hits = min_hits
        self.iou_threshold = iou_threshold; self.max_dist = max_dist
        self.tracks = []; self.track_id_counter = 1

    def update(self, dets):
        for t in self.tracks:
            t.predict()
        matched, unmatched_dets, unmatched_trks = self._assign(dets, self.tracks)
        for d, t in matched:
            self.tracks[t].update(dets[d])
        for t in unmatched_trks:
            self.tracks[t].miss()
        for d in unmatched_dets:
            self.tracks.append(Track(dets[d], self.track_id_counter))
            self.track_id_counter += 1
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
        return self.tracks

    def _assign(self, dets, tracks):
        if not tracks:
            return [], list(range(len(dets))), []
        if not dets:
            return [], [], list(range(len(tracks)))
        trk_boxes = [t.bbox for t in tracks]
        iou_mat = np.zeros((len(dets), len(trk_boxes)), np.float32)
        dist_mat = np.zeros((len(dets), len(trk_boxes)), np.float32)
        for d, det in enumerate(dets):
            for t, trk in enumerate(trk_boxes):
                iou_mat[d, t] = iou(det, trk); dist_mat[d, t] = center_distance(det, trk)
        ri, ci = linear_sum_assignment(-iou_mat)
        matched, md, mt = [], set(), set()
        for r, c in zip(ri, ci):
            if iou_mat[r, c] >= self.iou_threshold or dist_mat[r, c] <= self.max_dist:
                matched.append((r, c)); md.add(r); mt.add(c)
        return matched, [d for d in range(len(dets)) if d not in md], [t for t in range(len(tracks)) if t not in mt]


# ================================================================
# 4. Production detection params + post-processing helpers
# ================================================================

UNET_THRESHOLD = 0.95

PRODUCTION_DETECTION_PARAMS = {
    "diff_threshold": 24,
    "area_min": 12,
    "area_max": 856,
    "bbox_padding": 4,
    "max_age": 19,
    "min_hits": 5,
    "iou_thresh": 0.128,
    "max_dist": 52,
    "max_corners": 343,
    "lk_winsize": 17,
    "inlier_thresh": 0.578,
    "apply_platform_suppression": False,
}

LEGACY_PLATFORM_SUPPRESSION_ZONE_1280 = (0, 400, 175, 490)
LEGACY_PLATFORM_SUPPRESSION_REFERENCE_SIZE = (1280, 720)


def _prepare_unet_input(tensor, checkpoint_blur=False, out=None):
    source = np.asarray(tensor)[:3]
    expected_shape = tuple(source.shape)

    if (
        out is None and not checkpoint_blur
        and source.dtype == np.float32 and source.flags.c_contiguous
    ):
        return source

    if out is None:
        out = np.empty(expected_shape, dtype=np.float32)
    else:
        if out.shape != expected_shape:
            raise ValueError(f"Input buffer shape {out.shape} != expected {expected_shape}")
        if out.dtype != np.float32 or not out.flags.c_contiguous:
            raise ValueError("Input buffer must be C-contiguous float32")

    if checkpoint_blur:
        for channel_index in range(3):
            blurred = cv2.GaussianBlur(source[channel_index], (3, 3), 0)
            np.copyto(out[channel_index], blurred, casting="unsafe")
    else:
        np.copyto(out, source, casting="unsafe")

    return out


def _scaled_detection_geometry(height, width, params, return_size=(1280, 720)):
    reference_w, reference_h = 1280.0, 720.0
    scale_x = float(width) / reference_w
    scale_y = float(height) / reference_h
    area_scale = scale_x * scale_y

    area_min = max(1.0, float(params["area_min"]) * area_scale)
    area_max = max(area_min + 1.0, float(params["area_max"]) * area_scale)
    bbox_padding = max(0, int(round(float(params.get("bbox_padding", 4)) * scale_x)))

    return_w, return_h = map(int, return_size)
    return {
        "scale_x": scale_x, "scale_y": scale_y,
        "area_min": area_min, "area_max": area_max, "bbox_padding": bbox_padding,
        "output_scale_x": float(return_w) / float(width),
        "output_scale_y": float(return_h) / float(height),
    }


def _scale_reference_zone_to_mask(zone_xyxy, mask_shape, reference_size=LEGACY_PLATFORM_SUPPRESSION_REFERENCE_SIZE):
    height, width = map(int, mask_shape)
    reference_width, reference_height = map(int, reference_size)
    x1, y1, x2, y2 = map(float, zone_xyxy)

    scaled = [
        int(np.floor(x1 * width / max(1, reference_width))),
        int(np.floor(y1 * height / max(1, reference_height))),
        int(np.ceil(x2 * width / max(1, reference_width))),
        int(np.ceil(y2 * height / max(1, reference_height))),
    ]
    scaled[0] = int(np.clip(scaled[0], 0, width))
    scaled[2] = int(np.clip(scaled[2], 0, width))
    scaled[1] = int(np.clip(scaled[1], 0, height))
    scaled[3] = int(np.clip(scaled[3], 0, height))
    return tuple(scaled)


def _apply_platform_suppression_zone(binary_mask, zone_xyxy=LEGACY_PLATFORM_SUPPRESSION_ZONE_1280,
                                      reference_size=LEGACY_PLATFORM_SUPPRESSION_REFERENCE_SIZE):
    mask = np.asarray(binary_mask, dtype=np.uint8)
    x1, y1, x2, y2 = _scale_reference_zone_to_mask(zone_xyxy, mask.shape, reference_size=reference_size)
    if x2 > x1 and y2 > y1:
        mask[y1:y2, x1:x2] = 0
    return mask


def _probability_to_blobs(prob, tensor, threshold, area_min, area_max, bbox_padding,
                           apply_validity_mask=True, binary_mask=None,
                           platform_suppression_zone=None,
                           platform_reference_size=LEGACY_PLATFORM_SUPPRESSION_REFERENCE_SIZE):
    prob_array = None
    if prob is not None:
        prob_array = np.asarray(prob, dtype=np.float32)

    if binary_mask is None:
        if prob_array is None:
            raise ValueError("Either prob or binary_mask must be provided")
        thresh = (prob_array > float(threshold)).astype(np.uint8) * 255
    else:
        thresh = np.asarray(binary_mask, dtype=np.uint8)
        if thresh.ndim != 2:
            raise ValueError(f"binary_mask must be 2-D, got {thresh.shape}")
        thresh = np.ascontiguousarray(thresh)
        if thresh.size and int(thresh.max()) <= 1:
            thresh *= np.uint8(255)

    if apply_validity_mask:
        validity = np.asarray(tensor[3]) > 0.5
        if validity.shape != thresh.shape:
            raise ValueError(f"Validity shape {validity.shape} != mask shape {thresh.shape}")
        thresh[~validity] = 0

    if platform_suppression_zone is not None:
        _apply_platform_suppression_zone(thresh, zone_xyxy=platform_suppression_zone, reference_size=platform_reference_size)

    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_open, iterations=1)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    height, width = thresh.shape
    blobs = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if not (float(area_min) < area < float(area_max)):
            continue

        x, y, w, h = cv2.boundingRect(contour)
        if h <= 0 or (w / h) > 4.0:
            continue

        raw_bbox = [int(x), int(y), int(x + w), int(y + h)]
        padded_bbox = [
            max(0, int(x - bbox_padding)), max(0, int(y - bbox_padding)),
            min(width, int(x + w + bbox_padding)), min(height, int(y + h + bbox_padding)),
        ]

        if prob_array is None:
            prob_max = None
        else:
            region = prob_array[y:y + h, x:x + w]
            prob_max = float(region.max()) if region.size else 0.0

        blobs.append({"raw_bbox": raw_bbox, "bbox": padded_bbox, "area": area, "prob_max": prob_max})

    return thresh, blobs


# ================================================================
# 5. Production inference: video -> bbox dictionary
# ================================================================

def predict_video_bboxes(
    input_path, params, unet_model, unet_threshold=0.5, compensation_step=1,
    compensation_config=None, return_size=(1280, 720), apply_validity_mask=True,
    apply_platform_suppression=None, platform_suppression_zone=LEGACY_PLATFORM_SUPPRESSION_ZONE_1280,
    checkpoint_blur=False, diagnostics_callback=None,
):
    """
    Production inference path:

      video -> CompensationPipeline -> перші 3 канали -> TinyUNet
            -> ймовірність -> контурний фільтр -> SORT -> bbox по кадрах
    """
    if unet_model is None:
        raise ValueError("unet_model is required for the production path")
    if int(compensation_step) <= 0:
        raise ValueError("compensation_step must be positive")

    config = dict(PRODUCTION_CONFIG)
    if compensation_config is not None:
        config.update(dict(compensation_config))
    config["max_width"] = PROCESSING_MAX_WIDTH

    if apply_platform_suppression is None:
        apply_platform_suppression = bool(params.get("apply_platform_suppression", True))
    active_platform_zone = platform_suppression_zone if apply_platform_suppression else None

    pipeline = CompensationPipeline(config, step=int(compensation_step))
    unet_model.eval()

    tracker = None
    predictions = {}
    processing_shape = None
    geometry = None

    skipped_low_confidence = 0
    invalid_samples = 0
    input_buffer = None

    with torch.inference_mode():
        for sample in pipeline.stream_video(input_path):
            frame_idx = int(sample.frame_index) + 1

            if not sample.valid:
                invalid_samples += 1
                if tracker is not None:
                    tracker.update([])
                continue

            tensor = np.asarray(sample.tensor)
            if tensor.ndim != 3 or tensor.shape[0] != 4:
                raise RuntimeError(f"Unexpected classical tensor shape: {tensor.shape}")

            _, height, width = tensor.shape
            if height % 8 != 0 or width % 8 != 0:
                raise RuntimeError(f"TinyUNet requires H and W divisible by 8, got {(height, width)}")

            if tracker is None:
                processing_shape = (height, width)
                geometry = _scaled_detection_geometry(height, width, params, return_size=return_size)
                tracker = SortTracker(
                    max_age=params["max_age"], min_hits=params["min_hits"],
                    iou_threshold=params["iou_thresh"],
                    max_dist=max(1.0, float(params["max_dist"]) * geometry["scale_x"]),
                )
            elif processing_shape != (height, width):
                raise RuntimeError(f"Video changed processing size from {processing_shape} to {(height, width)}")

            min_ratio = params.get("inlier_thresh")
            if min_ratio is not None and float(sample.inlier_ratio) < float(min_ratio):
                skipped_low_confidence += 1
                tracker.update([])
                continue

            source_channels = tensor[:3]
            needs_copy_buffer = (
                checkpoint_blur or source_channels.dtype != np.float32 or not source_channels.flags.c_contiguous
            )
            if USE_PREALLOCATED_UNET_INPUT and needs_copy_buffer and input_buffer is None:
                input_buffer = np.empty((3, height, width), dtype=np.float32)

            x_np = _prepare_unet_input(
                tensor, checkpoint_blur=checkpoint_blur,
                out=(input_buffer if USE_PREALLOCATED_UNET_INPUT and needs_copy_buffer else None),
            )
            x_t = torch.from_numpy(x_np).unsqueeze(0).to(DEVICE, non_blocking=(DEVICE == "cuda"))

            logits = unet_model(x_t)
            probability_t = torch.sigmoid(logits)[0, 0]

            prob_u8_t = (probability_t > float(unet_threshold)).to(torch.uint8).mul_(255)
            thresh = prob_u8_t.cpu().numpy()

            prob = None
            if diagnostics_callback is not None:
                prob = probability_t.detach().cpu().numpy()

            thresh, blobs = _probability_to_blobs(
                prob=prob, binary_mask=thresh, tensor=tensor, threshold=unet_threshold,
                area_min=geometry["area_min"], area_max=geometry["area_max"],
                bbox_padding=geometry["bbox_padding"], apply_validity_mask=apply_validity_mask,
                platform_suppression_zone=active_platform_zone,
            )

            if diagnostics_callback is not None:
                diagnostics_callback({
                    "frame_index": frame_idx, "prob": prob, "binary_mask": thresh, "blobs": blobs,
                    "tensor": tensor, "inlier_ratio": float(sample.inlier_ratio),
                    "processing_shape": (height, width),
                })

            tracks = tracker.update([blob["bbox"] for blob in blobs])
            frame_predictions = []
            for track in tracks:
                if track.hits < params["min_hits"] or track.time_since_update != 0:
                    continue

                x1, y1, x2, y2 = track.bbox
                frame_predictions.append([
                    int(round(np.clip(x1, 0, width) * geometry["output_scale_x"])),
                    int(round(np.clip(y1, 0, height) * geometry["output_scale_y"])),
                    int(round(np.clip(x2, 0, width) * geometry["output_scale_x"])),
                    int(round(np.clip(y2, 0, height) * geometry["output_scale_y"])),
                ])

            if frame_predictions:
                predictions[frame_idx] = frame_predictions

    print("Classical pipeline stats:", pipeline.stats)
    print("Motion-estimation calls:", pipeline.motion_estimation_calls)
    print("Invalid/warmup samples:", invalid_samples)
    print("Skipped by inlier threshold:", skipped_low_confidence)
    return predictions


def predict_video_bboxes_pipelined(
    input_path, params, backend: ProbabilityBackend, unet_threshold=UNET_THRESHOLD,
    compensation_step=1, compensation_config=None, return_size=(1280, 720),
    apply_validity_mask=True, apply_platform_suppression=None,
    platform_suppression_zone=LEGACY_PLATFORM_SUPPRESSION_ZONE_1280,
    checkpoint_blur=False, queue_size=PIPELINE_QUEUE_SIZE,
    opencv_threads=OPENCV_THREADS, warmup_iterations=NEURAL_WARMUP_ITERATIONS,
    max_frames: Optional[int] = None, collect_timings: bool = False,
) -> dict:
    """
    Конвеєрний inference у два потоки.

        producer (CPU)  : decode -> CompensationPipeline -> тензор у слот
        consumer (GPU)  : backend.infer -> контурний фільтр -> SORT

    Залежність `класика N -> мережа N` зберігається повністю. Виграш у тому,
    що поки consumer рахує кадр N, producer уже рахує кадр N+1, тож
    пропускна здатність стає max(producer, consumer) замість їхньої суми.

    Ціна — +1 кадр затримки. Для підсвітки оператору це непомітно; для
    керуючого контуру треба лишати послідовний `predict_video_bboxes`.

    ВАЖЛИВО: `CompensationPipeline` перевикористовує внутрішній буфер, тому
    producer копіює тензор у слот з пулу. Без цієї копії consumer читав би
    дані, які producer уже перезаписав наступним кадром.
    """
    if backend is None:
        raise ValueError("backend is required (див. build_probability_backend)")
    if int(compensation_step) <= 0:
        raise ValueError("compensation_step must be positive")
    if int(queue_size) < 1:
        raise ValueError("queue_size must be >= 1")

    config = dict(PRODUCTION_CONFIG)
    if compensation_config is not None:
        config.update(dict(compensation_config))
    config["max_width"] = PROCESSING_MAX_WIDTH

    if apply_platform_suppression is None:
        apply_platform_suppression = bool(params.get("apply_platform_suppression", True))
    active_platform_zone = platform_suppression_zone if apply_platform_suppression else None

    predictions: Dict[int, list] = {}
    tracks_by_frame: Dict[int, list] = {}
    ready_queue: "queue.Queue[Optional[dict]]" = queue.Queue(maxsize=int(queue_size))
    free_slots: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=int(queue_size))
    stop_event = threading.Event()

    producer_state: Dict[str, Any] = {"error": None, "stats": None, "frames_seen": 0}
    classical_ms: List[float] = []
    inference_ms: List[float] = []
    postprocess_ms: List[float] = []
    latency_ms: List[float] = []
    stage_timings: List[dict] = []

    def producer() -> None:
        pipeline = CompensationPipeline(config, step=int(compensation_step))
        capture = cv2.VideoCapture(str(input_path))
        if not capture.isOpened():
            producer_state["error"] = FileNotFoundError(f"Could not open video: {input_path}")
            ready_queue.put(None)
            return

        source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        if bool(config["request_decoder_scaling"]):
            pipeline._decoder_target_size = _request_decoder_scaling(
                capture, int(config["max_width"]),
            )

        seen = 0
        try:
            while not stop_event.is_set():
                if max_frames is not None and seen >= int(max_frames):
                    break
                ok, frame = capture.read()
                if not ok:
                    break
                seen += 1

                if (
                    pipeline._decoder_target_size is not None
                    and source_width > int(config["max_width"])
                    and tuple(frame.shape[1::-1]) == tuple(pipeline._decoder_target_size)
                ):
                    pipeline.stats["decoder_scaled_frames"] += 1

                started = time.perf_counter()
                sample = pipeline.process_frame(frame)
                elapsed_ms = (time.perf_counter() - started) * 1000.0

                if not sample.valid:
                    # Невалідні кадри теж ідуть у чергу: consumer мусить
                    # постарити треки SORT, інакше вони "зависнуть".
                    ready_queue.put({
                        "valid": False, "frame_index": int(sample.frame_index),
                        "slot": None, "classical_ms": elapsed_ms,
                        "inlier_ratio": float(sample.inlier_ratio),
                        "ready_at": time.perf_counter(),
                    })
                    continue

                tensor = np.asarray(sample.tensor)
                slot = free_slots.get()
                if slot.shape != tensor.shape:
                    slot = np.empty(tensor.shape, dtype=np.float32)
                np.copyto(slot, tensor, casting="unsafe")

                ready_queue.put({
                    "valid": True, "frame_index": int(sample.frame_index),
                    "slot": slot, "classical_ms": elapsed_ms,
                    "inlier_ratio": float(sample.inlier_ratio),
                    "ready_at": time.perf_counter(),
                })
        except BaseException as error:  # noqa: BLE001 — передаємо в головний потік
            producer_state["error"] = error
        finally:
            capture.release()
            producer_state["stats"] = dict(pipeline.stats)
            producer_state["frames_seen"] = seen
            ready_queue.put(None)

    tracker = None
    geometry = None
    processing_shape = None
    invalid_samples = 0
    skipped_low_confidence = 0
    warmup_report = None
    warmed = False
    consumer_error: Optional[BaseException] = None

    # Пул слотів наповнюється ДО старту producer'а, інакше перший
    # free_slots.get() заблокується назавжди. Форма підбереться на
    # першому кадрі — producer перевиділить слот, якщо вона не збіглась.
    for _ in range(int(queue_size)):
        free_slots.put(np.empty((0, 0, 0), dtype=np.float32))

    with configure_opencv_threads(opencv_threads) as thread_info:
        producer_thread = threading.Thread(target=producer, name="classical-producer", daemon=True)
        producer_thread.start()

        try:
            while True:
                packet = ready_queue.get()
                if packet is None:
                    break

                if not packet["valid"]:
                    invalid_samples += 1
                    classical_ms.append(packet["classical_ms"])
                    if tracker is not None:
                        tracker.update([])
                    continue

                slot = packet["slot"]
                try:
                    _, height, width = slot.shape
                    if height % 8 != 0 or width % 8 != 0:
                        raise RuntimeError(
                            f"TinyUNet requires H and W divisible by 8, got {(height, width)}"
                        )

                    if tracker is None:
                        processing_shape = (height, width)
                        geometry = _scaled_detection_geometry(
                            height, width, params, return_size=return_size,
                        )
                        tracker = SortTracker(
                            max_age=params["max_age"], min_hits=params["min_hits"],
                            iou_threshold=params["iou_thresh"],
                            max_dist=max(1.0, float(params["max_dist"]) * geometry["scale_x"]),
                        )
                    elif processing_shape != (height, width):
                        raise RuntimeError(
                            f"Video changed processing size from {processing_shape} to {(height, width)}"
                        )

                    classical_ms.append(packet["classical_ms"])

                    min_ratio = params.get("inlier_thresh")
                    if min_ratio is not None and float(packet["inlier_ratio"]) < float(min_ratio):
                        skipped_low_confidence += 1
                        tracker.update([])
                        continue

                    source = _prepare_unet_input(slot, checkpoint_blur=checkpoint_blur, out=None)

                    if not warmed:
                        # Прогрів ДО першого заміру: компіляція шейдерів MPS
                        # або графа Core ML не має потрапити в статистику.
                        warmup_report = backend.warmup(
                            source, unet_threshold, iterations=warmup_iterations,
                        )
                        warmed = True
                        # Перештампувати ready_at, інакше вартість прогріву
                        # осіла б у latency саме цього кадру.
                        packet["ready_at"] = time.perf_counter()

                    if collect_timings:
                        mask, timings = backend.infer_profiled(source, unet_threshold)
                        stage_timings.append(timings)
                        inference_ms.append(timings["total_ms"])
                    else:
                        started = time.perf_counter()
                        mask = backend.infer(source, unet_threshold)
                        inference_ms.append((time.perf_counter() - started) * 1000.0)

                    post_started = time.perf_counter()
                    _, blobs = _probability_to_blobs(
                        prob=None, binary_mask=mask, tensor=slot, threshold=unet_threshold,
                        area_min=geometry["area_min"], area_max=geometry["area_max"],
                        bbox_padding=geometry["bbox_padding"],
                        apply_validity_mask=apply_validity_mask,
                        platform_suppression_zone=active_platform_zone,
                    )

                    tracks = tracker.update([blob["bbox"] for blob in blobs])
                    frame_predictions = []
                    frame_track_records = []
                    for track in tracks:
                        if track.hits < params["min_hits"] or track.time_since_update != 0:
                            continue
                        x1, y1, x2, y2 = track.bbox
                        scaled = [
                            float(np.clip(x1, 0, width) * geometry["output_scale_x"]),
                            float(np.clip(y1, 0, height) * geometry["output_scale_y"]),
                            float(np.clip(x2, 0, width) * geometry["output_scale_x"]),
                            float(np.clip(y2, 0, height) * geometry["output_scale_y"]),
                        ]
                        frame_predictions.append([int(round(v)) for v in scaled])
                        # Float-координати + track_id потрібні лише для офлайн
                        # згладжування (presentation.py). `predictions` лишається
                        # байт-у-байт таким, як був, щоб не зачепити метрики.
                        frame_track_records.append({
                            "track_id": int(track.track_id),
                            "bbox": scaled,
                            "hits": int(track.hits),
                        })
                    frame_number = int(packet["frame_index"]) + 1
                    if frame_predictions:
                        predictions[frame_number] = frame_predictions
                        tracks_by_frame[frame_number] = frame_track_records

                    postprocess_ms.append((time.perf_counter() - post_started) * 1000.0)
                    latency_ms.append((time.perf_counter() - packet["ready_at"]) * 1000.0)
                finally:
                    # Слот повертається навіть при continue чи винятку,
                    # інакше пул вичерпається і producer зависне назавжди.
                    free_slots.put(slot)

        except BaseException as error:  # noqa: BLE001
            consumer_error = error
            stop_event.set()
        finally:
            if consumer_error is not None:
                # Розвантажуємо чергу, щоб producer не заблокувався на put().
                while producer_thread.is_alive():
                    try:
                        item = ready_queue.get(timeout=0.05)
                    except queue.Empty:
                        continue
                    if item is not None and item.get("slot") is not None:
                        free_slots.put(item["slot"])
            producer_thread.join(timeout=10.0)

    if consumer_error is not None:
        raise consumer_error
    if producer_state["error"] is not None:
        raise producer_state["error"]

    def summarize(values):
        if not values:
            return {"mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "fps": 0.0, "n": 0}
        arr = np.asarray(values, dtype=np.float64)
        mean_ms = float(arr.mean())
        return {
            "mean_ms": mean_ms,
            "p50_ms": float(np.percentile(arr, 50)),
            "p95_ms": float(np.percentile(arr, 95)),
            "fps": 1000.0 / mean_ms if mean_ms > 0 else 0.0,
            "n": int(len(values)),
        }

    producer_summary = summarize(classical_ms)
    consumer_ms = [i + p for i, p in zip(inference_ms, postprocess_ms)]
    consumer_summary = summarize(consumer_ms)
    bottleneck_ms = max(producer_summary["mean_ms"], consumer_summary["mean_ms"])

    return {
        "predictions": predictions,
        "tracks_by_frame": tracks_by_frame,
        "backend": backend.name,
        "device": DEVICE,
        "processing_shape_hw": processing_shape,
        "queue_size": int(queue_size),
        "opencv_threads": dict(thread_info),
        "warmup": warmup_report,
        "frames_seen": producer_state["frames_seen"],
        "invalid_samples": invalid_samples,
        "skipped_low_confidence": skipped_low_confidence,
        "classical_stats": producer_state["stats"],
        "producer_per_frame": producer_summary,
        "consumer_per_frame": consumer_summary,
        "neural_network": summarize(inference_ms),
        "postprocessing_sort": summarize(postprocess_ms),
        "end_to_end_latency": summarize(latency_ms),
        "stage_timings": stage_timings if collect_timings else None,
        "pipelined_ms_per_frame": bottleneck_ms,
        "pipelined_fps": 1000.0 / bottleneck_ms if bottleneck_ms > 0 else 0.0,
        "sequential_ms_per_frame": producer_summary["mean_ms"] + consumer_summary["mean_ms"],
        "bottleneck": (
            "producer" if producer_summary["mean_ms"] >= consumer_summary["mean_ms"] else "consumer"
        ),
    }


def benchmark_opencv_threads_ab(
    video_path, backend_factory, params=PRODUCTION_DETECTION_PARAMS,
    unet_threshold=UNET_THRESHOLD, max_frames: Optional[int] = 300,
    thread_options=(1, 0), queue_size=PIPELINE_QUEUE_SIZE,
) -> dict:
    """
    A/B по кількості потоків OpenCV на КОНВЕЄРНОМУ шляху.

    Питання, яке це закриває: чи внутрішній пул OpenCV у producer'і
    конкурує з consumer'ом за ядра. На послідовному шляху відповіді немає.

    `backend_factory` викликається заново для кожної конфігурації, щоб
    друга не успадкувала вже прогрітий бекенд від першої.
    """
    results = {}
    for threads in thread_options:
        backend = backend_factory()
        try:
            report = predict_video_bboxes_pipelined(
                video_path, params, backend, unet_threshold=unet_threshold,
                max_frames=max_frames, opencv_threads=threads,
                queue_size=queue_size, collect_timings=True,
            )
        finally:
            backend.close()
        report.pop("predictions", None)
        report.pop("stage_timings", None)
        results[str(threads)] = report

    best = min(results.items(), key=lambda item: item[1]["pipelined_ms_per_frame"])
    return {
        "runs": results,
        "best_threads": best[0],
        "best_ms_per_frame": best[1]["pipelined_ms_per_frame"],
        "best_fps": best[1]["pipelined_fps"],
    }


def render_predictions_video(input_path, output_path, predictions, output_size=(1280, 720),
                              box_color=(0, 255, 0), box_thickness=2):
    """Записує MP4 із bbox, які повернув predict_video_bboxes()."""
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Не вдалося відкрити відео: {input_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        fps = 25.0

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_w, out_h = map(int, output_size)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, out_h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Не вдалося створити відео: {output_path}")

    frame_index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_index += 1
            frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)

            boxes = predictions.get(frame_index, [])
            for x1, y1, x2, y2 in boxes:
                x1 = int(np.clip(x1, 0, out_w - 1))
                y1 = int(np.clip(y1, 0, out_h - 1))
                x2 = int(np.clip(x2, 0, out_w - 1))
                y2 = int(np.clip(y2, 0, out_h - 1))
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, box_thickness)

            cv2.putText(
                frame, f"frame={frame_index} detections={len(boxes)}", (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA,
            )
            writer.write(frame)
    finally:
        cap.release()
        writer.release()

    print(f"Відео збережено: {output_path}")
    return str(output_path)


# ================================================================
# 6. ПРОФІЛЮВАННЯ: FLOPs нейромережі + окрема швидкість
#    класики і нейромережі
# ================================================================
#
# Класична частина (GFTT, forward/backward LK, RANSAC, warpPerspective,
# морфологія, findContours) НЕ має стандартного означення "FLOPs" — це
# не матрична арифметика на кшталт згорток, а суміш ітеративних
# алгоритмів OpenCV зі змінною кількістю ітерацій. Тому для неї
# коректно вимірювати саме РЕАЛЬНИЙ ЧАС (ms/frame, FPS), а не FLOPs.
#
# Для TinyUNet, навпаки, FLOPs добре визначені (кожен Conv2d — це набір
# MAC-операцій), тож рахуємо їх точно через forward-хуки.


def count_model_flops(model: nn.Module, input_shape: Tuple[int, int, int, int]) -> dict:
    """
    Точно рахує FLOPs/MACs для одного forward-проходу TinyUNet при
    заданій формі входу (B, C, H, W).

    MACs (multiply-accumulate) для Conv2d:
        MACs = Cout * (Cin / groups) * Kh * Kw * Hout * Wout
    FLOPs (прийнято рахувати множення + додавання окремо):
        FLOPs = 2 * MACs
    """
    macs_total = {"value": 0}
    hooks = []

    def conv_hook(module: nn.Conv2d, inputs, output):
        out_h, out_w = output.shape[-2:]
        cin_per_group = module.in_channels // module.groups
        kh, kw = module.kernel_size
        macs = module.out_channels * cin_per_group * kh * kw * out_h * out_w
        macs_total["value"] += int(macs)

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(conv_hook))

    model.eval()
    first_parameter = next(model.parameters(), None)
    target_device = first_parameter.device if first_parameter is not None else torch.device("cpu")
    dummy = torch.zeros(input_shape, dtype=torch.float32, device=target_device)
    with torch.inference_mode():
        model(dummy)

    for hook in hooks:
        hook.remove()

    macs = macs_total["value"]
    flops = 2 * macs
    n_params = sum(p.numel() for p in model.parameters())

    return {
        "input_shape": tuple(input_shape),
        "macs": macs,
        "flops": flops,
        "gflops": flops / 1e9,
        "params": n_params,
        "note": (
            "FLOPs рахуються тільки для Conv2d-шарів (це основна маса роботи "
            "TinyUNet). Внесок F.interpolate (bilinear upsample) є на 1-2 "
            "порядки меншим і тут не враховується."
        ),
    }


def benchmark_pipeline(
    video_path, unet_model=None, params=PRODUCTION_DETECTION_PARAMS, unet_threshold=UNET_THRESHOLD,
    compensation_step=1, max_frames: Optional[int] = None,
    warmup_frames: int = NEURAL_WARMUP_ITERATIONS, backend: Optional[ProbabilityBackend] = None,
    opencv_threads=OPENCV_THREADS,
) -> dict:
    """
    Послідовний бенчмарк зі СТАДІЙНОЮ розбивкою.

    Раніше `_prepare_unet_input` + H2D стояли до старту таймера, а D2H —
    після його зупинки, тож обидва трансфери були невидимі й спливали
    лише в end-to-end. Тепер вимірюються всі шість стадій, і саме
    співвідношення transfer/compute вирішує, чи окупиться fp16 та ANE.

    Перші `warmup_frames` валідних кадрів виключаються зі статистики.
    """
    if backend is None:
        if unet_model is None:
            raise ValueError("Потрібен unet_model або готовий backend")
        backend = TorchProbabilityBackend(unet_model, device=DEVICE)

    config = dict(PRODUCTION_CONFIG)
    config["max_width"] = PROCESSING_MAX_WIDTH

    pipeline = CompensationPipeline(config, step=int(compensation_step))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    classical_times, prepare_times = [], []
    h2d_times, compute_times, d2h_times = [], [], []
    nn_times, postproc_times, end_to_end_times = [], [], []

    flops_report = None
    tracker = None
    geometry = None
    processing_shape = None
    input_buffer = None
    warmup_report = None
    warmed = False

    n_seen = 0
    n_valid = 0

    with configure_opencv_threads(opencv_threads) as thread_info:
        try:
            while True:
                if max_frames is not None and n_seen >= int(max_frames):
                    break
                ok, frame = cap.read()
                if not ok:
                    break
                n_seen += 1

                frame_start = time.perf_counter()

                # ---- Класика ----
                t0 = time.perf_counter()
                sample = pipeline.process_frame(frame)
                t1 = time.perf_counter()

                if not sample.valid:
                    if tracker is not None:
                        tracker.update([])
                    continue

                n_valid += 1
                tensor = np.asarray(sample.tensor)
                _, height, width = tensor.shape

                if tracker is None:
                    processing_shape = (height, width)
                    geometry = _scaled_detection_geometry(height, width, params, return_size=(1280, 720))
                    tracker = SortTracker(
                        max_age=params["max_age"], min_hits=params["min_hits"],
                        iou_threshold=params["iou_thresh"],
                        max_dist=max(1.0, float(params["max_dist"]) * geometry["scale_x"]),
                    )
                    if isinstance(backend, TorchProbabilityBackend):
                        flops_report = count_model_flops(backend.model, (1, 3, height, width))

                # ---- Підготовка входу (та сама гілка, що в продакшені) ----
                t2 = time.perf_counter()
                source_channels = tensor[:3]
                needs_copy_buffer = (
                    source_channels.dtype != np.float32 or not source_channels.flags.c_contiguous
                )
                if USE_PREALLOCATED_UNET_INPUT and needs_copy_buffer and input_buffer is None:
                    input_buffer = np.empty((3, height, width), dtype=np.float32)
                x_np = _prepare_unet_input(
                    tensor, checkpoint_blur=False,
                    out=(input_buffer if USE_PREALLOCATED_UNET_INPUT and needs_copy_buffer else None),
                )
                t3 = time.perf_counter()

                if not warmed:
                    warmup_report = backend.warmup(x_np, unet_threshold, iterations=warmup_frames)
                    warmed = True

                # ---- Нейромережа: h2d / compute / d2h ----
                thresh, stage = backend.infer_profiled(x_np, unet_threshold)

                # ---- Постобробка ----
                t4 = time.perf_counter()
                _, blobs = _probability_to_blobs(
                    prob=None, binary_mask=thresh, tensor=tensor, threshold=unet_threshold,
                    area_min=geometry["area_min"], area_max=geometry["area_max"],
                    bbox_padding=geometry["bbox_padding"], apply_validity_mask=True,
                    platform_suppression_zone=None,
                )
                tracker.update([blob["bbox"] for blob in blobs])
                t5 = time.perf_counter()

                if n_valid > warmup_frames:
                    classical_times.append((t1 - t0) * 1000.0)
                    prepare_times.append((t3 - t2) * 1000.0)
                    h2d_times.append(stage["h2d_ms"])
                    compute_times.append(stage["compute_ms"])
                    d2h_times.append(stage["d2h_ms"])
                    nn_times.append(stage["total_ms"])
                    postproc_times.append((t5 - t4) * 1000.0)
                    end_to_end_times.append((t5 - frame_start) * 1000.0)
        finally:
            cap.release()

    def summarize(values_ms):
        clean = [v for v in values_ms if v == v]  # відкидаємо NaN (Core ML)
        if not clean:
            return {"mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "fps": 0.0, "n": 0}
        arr = np.asarray(clean, dtype=np.float64)
        mean_ms = float(arr.mean())
        return {
            "mean_ms": mean_ms,
            "p50_ms": float(np.percentile(arr, 50)),
            "p95_ms": float(np.percentile(arr, 95)),
            "fps": 1000.0 / mean_ms if mean_ms > 0 else 0.0,
            "n": int(len(clean)),
        }

    nn_summary = summarize(nn_times)
    achieved_gflops_per_s = None
    if flops_report is not None and nn_summary["fps"] > 0:
        achieved_gflops_per_s = flops_report["gflops"] * nn_summary["fps"]

    return {
        "device": DEVICE,
        "backend": backend.name,
        "opencv_threads": dict(thread_info),
        "processing_shape_hw": processing_shape,
        "frames_seen": n_seen,
        "frames_valid": n_valid,
        "warmup_frames_excluded": warmup_frames,
        "warmup": warmup_report,
        "classical_compensation": summarize(classical_times),
        "input_prepare": summarize(prepare_times),
        "h2d_transfer": summarize(h2d_times),
        "gpu_compute": summarize(compute_times),
        "d2h_transfer": summarize(d2h_times),
        "neural_network": nn_summary,
        "postprocessing_sort": summarize(postproc_times),
        "end_to_end_per_frame": summarize(end_to_end_times),
        "unet_flops_per_frame": flops_report,
        "achieved_gflops_per_second": achieved_gflops_per_s,
    }


def print_benchmark_report(report: dict) -> None:
    print("\n" + "=" * 72)
    print("BENCHMARK REPORT (послідовний шлях, стадійна розбивка)")
    print("=" * 72)
    print(f"Device / backend:       {report['device']} / {report.get('backend')}")
    threads = report.get("opencv_threads") or {}
    print(
        f"OpenCV threads:         requested={threads.get('requested')} "
        f"actual={threads.get('actual')} restored={threads.get('restored_to')}"
    )
    print(f"Processing shape (HxW): {report['processing_shape_hw']}")
    print(f"Frames seen / valid:    {report['frames_seen']} / {report['frames_valid']}")
    print(f"Warmup excluded:        {report['warmup_frames_excluded']} кадрів")
    if report.get("warmup"):
        print(f"Backend warmup:         {report['warmup']['total_ms']:.1f} ms "
              f"({report['warmup']['iterations']} ітерацій)")

    def line(name, summary):
        if not summary or summary["n"] == 0:
            print(f"{name:<30} —")
            return
        print(
            f"{name:<30} mean={summary['mean_ms']:.3f}  "
            f"p50={summary.get('p50_ms', 0):.3f}  p95={summary.get('p95_ms', 0):.3f} ms  "
            f"FPS={summary['fps']:.1f}  (n={summary['n']})"
        )

    print("-" * 72)
    line("1 класика (compensation)", report["classical_compensation"])
    line("2 підготовка входу", report.get("input_prepare"))
    line("3 h2d transfer", report.get("h2d_transfer"))
    line("4 gpu compute", report.get("gpu_compute"))
    line("5 d2h transfer", report.get("d2h_transfer"))
    line("  нейромережа (3+4+5)", report["neural_network"])
    line("6 постобробка + SORT", report["postprocessing_sort"])
    line("  end-to-end на кадр", report["end_to_end_per_frame"])
    print("-" * 72)

    if str(report.get("backend", "")).startswith("coreml"):
        print("Примітка: Core ML не розділяє transfer і compute — стадії 3/5 порожні,")
        print("          весь час зведений у 'gpu compute'.")

    flops = report["unet_flops_per_frame"]
    if flops is not None:
        print(f"TinyUNet FLOPs / кадр:  {flops['gflops']:.4f} GFLOPs  (params={flops['params']:,})")
        if report["achieved_gflops_per_second"] is not None:
            print(f"Досягнута продуктивність NN: {report['achieved_gflops_per_second']:.2f} GFLOPS/с")
    print("=" * 72)


def print_pipelined_report(report: dict) -> None:
    print("\n" + "=" * 72)
    print("PIPELINED REPORT (два потоки: CPU-класика ∥ GPU-мережа)")
    print("=" * 72)
    print(f"Backend:                {report['backend']}")
    threads = report.get("opencv_threads") or {}
    print(
        f"OpenCV threads:         requested={threads.get('requested')} "
        f"actual={threads.get('actual')} restored={threads.get('restored_to')}"
    )
    print(f"Queue size:             {report['queue_size']}")
    print(f"Processing shape (HxW): {report['processing_shape_hw']}")
    print(f"Frames seen:            {report['frames_seen']}")
    print(f"Invalid / low-conf:     {report['invalid_samples']} / {report['skipped_low_confidence']}")
    if report.get("warmup"):
        print(f"Backend warmup:         {report['warmup']['total_ms']:.1f} ms "
              f"({report['warmup']['iterations']} ітерацій)")

    def line(name, summary):
        if not summary or summary["n"] == 0:
            print(f"{name:<30} —")
            return
        print(
            f"{name:<30} mean={summary['mean_ms']:.3f}  "
            f"p95={summary.get('p95_ms', 0):.3f} ms  FPS={summary['fps']:.1f}  (n={summary['n']})"
        )

    print("-" * 72)
    line("producer (класика, CPU)", report["producer_per_frame"])
    line("consumer (мережа+пост)", report["consumer_per_frame"])
    line("  з них нейромережа", report["neural_network"])
    line("  з них постобробка", report["postprocessing_sort"])
    line("latency в consumer'і", report["end_to_end_latency"])
    print("-" * 72)
    print(f"Вузьке місце:           {report['bottleneck']}")
    print(f"Послідовно було б:      {report['sequential_ms_per_frame']:.3f} ms/frame")
    print(
        f"Конвеєром:              {report['pipelined_ms_per_frame']:.3f} ms/frame  "
        f"({report['pipelined_fps']:.1f} FPS)"
    )
    if report["pipelined_ms_per_frame"] > 0:
        speedup = report["sequential_ms_per_frame"] / report["pipelined_ms_per_frame"]
        print(f"Приріст від конвеєра:   {speedup:.2f}×")
    print("Нагадування: конвеєр додає ~1 кадр затримки. Для керуючого контуру")
    print("             використовуйте послідовний predict_video_bboxes.")
    print("=" * 72)


# ================================================================
# 7. CLI
# ================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="TinyUNet inference for small moving object detection.")
    parser.add_argument("--video", required=True, help="Шлях до вхідного відео.")
    parser.add_argument("--checkpoint", default=None, help="Шлях до .pth checkpoint TinyUNet.")
    parser.add_argument("--coreml", default=None, help="Шлях до .mlpackage — вмикає Core ML бекенд.")
    parser.add_argument("--compute-units", default="ALL", choices=["ALL", "CPU_AND_GPU", "CPU_AND_NE", "CPU_ONLY"],
                        help="Core ML: які блоки задіяти. ALL дозволяє ANE.")
    parser.add_argument("--output", default=None, help="Шлях для відео з відмальованими bbox.")
    parser.add_argument("--predictions-json", default=None, help="Шлях для збереження predictions у JSON.")
    parser.add_argument("--tracks-json", default=None, help="Шлях для збереження tracks_by_frame з track_id для presentation.py. Працює у --pipelined режимі.")
    parser.add_argument("--unet-threshold", type=float, default=UNET_THRESHOLD)
    parser.add_argument("--max-frames", type=int, default=None, help="Обмежити кількість кадрів.")

    parser.add_argument("--pipelined", action="store_true",
                        help="Two-thread pipelined inference for higher throughput.")
    parser.add_argument("--queue-size", type=int, default=PIPELINE_QUEUE_SIZE)
    parser.add_argument("--opencv-threads", type=int, default=OPENCV_THREADS,
                        help="Потоки OpenCV: 1 — рекомендовано для конвеєра, 0 — на розсуд бібліотеки.")
    parser.add_argument("--warmup-frames", type=int, default=NEURAL_WARMUP_ITERATIONS,
                        help="Ітерацій прогріву бекенду перед вимірюванням.")

    parser.add_argument("--benchmark", action="store_true", help="Послідовний бенчмарк зі стадіями.")
    parser.add_argument("--benchmark-pipelined", action="store_true", help="Бенчмарк конвеєра.")
    parser.add_argument("--threads-ab", action="store_true",
                        help="A/B по потоках OpenCV на конвеєрі (закриває питання контенції).")

    parser.add_argument("--export-coreml", default=None,
                        help="Експортувати checkpoint у Core ML fp16 за вказаним шляхом і вийти.")
    parser.add_argument("--export-height", type=int, default=360)
    parser.add_argument("--export-width", type=int, default=PROCESSING_MAX_WIDTH)
    parser.add_argument("--verify-parity", action="store_true",
                        help="Після експорту звірити Core ML fp16 з PyTorch fp32 на реальних кадрах.")
    args = parser.parse_args()

    if args.export_coreml:
        if not args.checkpoint:
            parser.error("--export-coreml вимагає --checkpoint")
        info = export_tinyunet_coreml_fp16(
            args.checkpoint, args.export_coreml,
            input_shape=(1, 3, args.export_height, args.export_width),
            compute_units=args.compute_units,
        )
        print("Core ML експортовано:", json.dumps(info, ensure_ascii=False, indent=2))
        if args.verify_parity:
            torch_backend = TorchProbabilityBackend(
                load_frozen_pytorch_model(args.checkpoint, device=DEVICE), device=DEVICE,
            )
            coreml_backend = CoreMLProbabilityBackend(args.export_coreml, compute_units=args.compute_units)
            parity = verify_coreml_parity(torch_backend, coreml_backend, args.video)
            print("Parity fp16 vs fp32:", json.dumps(parity, ensure_ascii=False, indent=2))
        return

    if not args.checkpoint and not args.coreml:
        parser.error("Потрібен --checkpoint або --coreml")

    print("Device:", DEVICE)

    def make_backend() -> ProbabilityBackend:
        return build_probability_backend(
            checkpoint_path=args.checkpoint, coreml_path=args.coreml,
            device=DEVICE, compute_units=args.compute_units,
        )

    if args.threads_ab:
        result = benchmark_opencv_threads_ab(
            args.video, make_backend, unet_threshold=args.unet_threshold,
            max_frames=args.max_frames or 300, queue_size=args.queue_size,
        )
        for threads, report in result["runs"].items():
            print(f"\n### OPENCV_THREADS = {threads}")
            print_pipelined_report(report)
        print(f"\nНайкраще: OPENCV_THREADS={result['best_threads']} -> "
              f"{result['best_ms_per_frame']:.3f} ms/frame ({result['best_fps']:.1f} FPS)")
        return

    backend = make_backend()
    print("Backend:", backend.name)
    tracks_by_frame = {}

    try:
        if args.benchmark:
            report = benchmark_pipeline(
                args.video, backend=backend, params=PRODUCTION_DETECTION_PARAMS,
                unet_threshold=args.unet_threshold, max_frames=args.max_frames,
                warmup_frames=args.warmup_frames, opencv_threads=args.opencv_threads,
            )
            print_benchmark_report(report)
            return

        if args.pipelined or args.benchmark_pipelined:
            report = predict_video_bboxes_pipelined(
                args.video, PRODUCTION_DETECTION_PARAMS, backend,
                unet_threshold=args.unet_threshold, queue_size=args.queue_size,
                opencv_threads=args.opencv_threads, warmup_iterations=args.warmup_frames,
                max_frames=args.max_frames, collect_timings=args.benchmark_pipelined,
            )
            predictions = report.pop("predictions")
            tracks_by_frame = report.pop("tracks_by_frame", {})
            print_pipelined_report(report)
        else:
            if not isinstance(backend, TorchProbabilityBackend):
                parser.error("Послідовний predict_video_bboxes працює лише з PyTorch. "
                             "Для Core ML використайте --pipelined.")
            with configure_opencv_threads(args.opencv_threads):
                predictions = predict_video_bboxes(
                    args.video, PRODUCTION_DETECTION_PARAMS, backend.model,
                    unet_threshold=args.unet_threshold,
                )
    finally:
        backend.close()

    if args.tracks_json:
        if not (args.pipelined or args.benchmark_pipelined):
            parser.error("--tracks-json вимагає --pipelined або --benchmark-pipelined")
        Path(args.tracks_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.tracks_json, "w", encoding="utf-8") as f:
            json.dump(tracks_by_frame, f, ensure_ascii=False, indent=2)
        print("Tracks saved:", args.tracks_json)

    if args.predictions_json:
        Path(args.predictions_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.predictions_json, "w", encoding="utf-8") as f:
            json.dump(predictions, f, ensure_ascii=False, indent=2)
        print("Predictions saved:", args.predictions_json)

    if args.output:
        render_predictions_video(args.video, args.output, predictions)


if __name__ == "__main__":
    main()