"""Quantitative comparison of a reconstruction with its canonical source."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class VisualScore:
    structural_similarity: float
    edge_f1: float
    color_similarity: float
    overall: float

    def as_dict(self) -> dict[str, float]:
        return {
            "structural_similarity": self.structural_similarity,
            "edge_f1": self.edge_f1,
            "color_similarity": self.color_similarity,
            "overall": self.overall,
        }

def _rgb_array(image: Image.Image, size: tuple[int, int]) -> np.ndarray:
    return np.asarray(image.convert("RGB").resize(size, Image.Resampling.LANCZOS))


def _ssim(gray_a: np.ndarray, gray_b: np.ndarray) -> float:
    a = gray_a.astype(np.float32)
    b = gray_b.astype(np.float32)
    mean_a = cv2.GaussianBlur(a, (11, 11), 1.5)
    mean_b = cv2.GaussianBlur(b, (11, 11), 1.5)
    variance_a = cv2.GaussianBlur(a * a, (11, 11), 1.5) - mean_a * mean_a
    variance_b = cv2.GaussianBlur(b * b, (11, 11), 1.5) - mean_b * mean_b
    covariance = cv2.GaussianBlur(a * b, (11, 11), 1.5) - mean_a * mean_b
    c1, c2 = 6.5025, 58.5225
    numerator = (2 * mean_a * mean_b + c1) * (2 * covariance + c2)
    denominator = (mean_a * mean_a + mean_b * mean_b + c1) * (
        variance_a + variance_b + c2
    )
    return float(np.clip(np.mean(numerator / np.maximum(denominator, 1e-6)), 0, 1))


def _edge_f1(gray_a: np.ndarray, gray_b: np.ndarray) -> float:
    edge_a = cv2.Canny(gray_a, 60, 160) > 0
    edge_b = cv2.Canny(gray_b, 60, 160) > 0
    kernel = np.ones((3, 3), np.uint8)
    near_a = cv2.dilate(edge_a.astype(np.uint8), kernel) > 0
    near_b = cv2.dilate(edge_b.astype(np.uint8), kernel) > 0
    precision = float(np.count_nonzero(edge_b & near_a)) / max(1, np.count_nonzero(edge_b))
    recall = float(np.count_nonzero(edge_a & near_b)) / max(1, np.count_nonzero(edge_a))
    return 2 * precision * recall / max(1e-6, precision + recall)


def compare_images(source: Image.Image, reconstruction: Image.Image) -> VisualScore:
    """Compare at a bounded common resolution for stable, fast diagnostics."""
    width = min(1400, source.width)
    height = max(1, round(width * source.height / source.width))
    size = (width, height)
    source_rgb = _rgb_array(source, size)
    output_rgb = _rgb_array(reconstruction, size)
    source_gray = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2GRAY)
    output_gray = cv2.cvtColor(output_rgb, cv2.COLOR_RGB2GRAY)
    structural = _ssim(source_gray, output_gray)
    edge = _edge_f1(source_gray, output_gray)
    source_lab = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    output_lab = cv2.cvtColor(output_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    mean_delta = float(np.mean(np.linalg.norm(source_lab - output_lab, axis=2)))
    color = float(np.exp(-mean_delta / 18.0))
    overall = 0.55 * structural + 0.35 * edge + 0.10 * color
    return VisualScore(structural, edge, color, overall)
def difference_heatmap(source: Image.Image, reconstruction: Image.Image) -> Image.Image:
    output = reconstruction.convert("RGB").resize(source.size, Image.Resampling.LANCZOS)
    a = np.asarray(source.convert("RGB"), dtype=np.int16)
    b = np.asarray(output, dtype=np.int16)
    difference = np.mean(np.abs(a - b), axis=2).astype(np.uint8)
    heat = cv2.applyColorMap(cv2.normalize(difference, None, 0, 255, cv2.NORM_MINMAX), cv2.COLORMAP_TURBO)
    return Image.fromarray(cv2.cvtColor(heat, cv2.COLOR_BGR2RGB))
