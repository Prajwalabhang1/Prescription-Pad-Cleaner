"""Document rectification and high-fidelity restoration."""

from __future__ import annotations

from dataclasses import dataclass
import io
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps


MAX_ANALYSIS_DIMENSION = 2048
PRINT_LONG_EDGE_PX = 3508  # A4 long edge at 300 DPI.


@dataclass(frozen=True)
class PreprocessResult:
    """Images and geometry produced from one uploaded camera image or scan."""

    canonical: Image.Image
    restored: Image.Image
    analysis: Image.Image
    source_size: tuple[int, int]
    page_corners: Optional[tuple[tuple[float, float], ...]]
    page_confidence: float = 0.0
    page_detection_method: str = "full-image"

    @property
    def page_size(self) -> tuple[int, int]:
        return self.canonical.size

    @property
    def needs_manual_crop(self) -> bool:
        return self.page_confidence < 0.72


@dataclass(frozen=True)
class PageDetection:
    corners: np.ndarray
    confidence: float
    method: str


def _decode(image_bytes: bytes) -> tuple[np.ndarray, tuple[int, int]]:
    try:
        with Image.open(io.BytesIO(image_bytes)) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            size = image.size
            return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR), size
    except Exception as exc:
        raise ValueError("Could not decode image") from exc


def _to_cv2(image_bytes: bytes) -> np.ndarray:
    """Backward-compatible decoder used by older callers and tests."""
    return _decode(image_bytes)[0]


def image_size(image_bytes: bytes) -> tuple[int, int]:
    """Return display dimensions after applying a camera's EXIF orientation."""
    return _decode(image_bytes)[1]


def _order_corners(points: np.ndarray) -> np.ndarray:
    points = points.astype(np.float32).reshape(4, 2)
    ordered = np.zeros((4, 2), dtype=np.float32)
    sums = points.sum(axis=1)
    differences = np.diff(points, axis=1).reshape(-1)
    ordered[0] = points[np.argmin(sums)]  # top-left
    ordered[2] = points[np.argmax(sums)]  # bottom-right
    ordered[1] = points[np.argmin(differences)]  # top-right
    ordered[3] = points[np.argmax(differences)]  # bottom-left
    return ordered


def _candidate_quadrilateral(mask: np.ndarray) -> Optional[np.ndarray]:
    height, width = mask.shape[:2]
    minimum_area = height * width * 0.28
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:12]:
        area = cv2.contourArea(contour)
        if area < minimum_area:
            continue
        perimeter = cv2.arcLength(contour, True)
        approximation = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approximation) == 4 and cv2.isContourConvex(approximation):
            return _order_corners(approximation)
        rectangle = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rectangle)
        box_area = max(1.0, rectangle[1][0] * rectangle[1][1])
        if area / box_area >= 0.72:
            return _order_corners(box)
    return None


def _contour_quad_candidates(mask: np.ndarray) -> list[np.ndarray]:
    """Return viable four-corner page candidates from one segmentation mask."""
    height, width = mask.shape[:2]
    minimum_area = height * width * 0.20
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[np.ndarray] = []
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:10]:
        if cv2.contourArea(contour) < minimum_area:
            continue
        perimeter = cv2.arcLength(contour, True)
        approximation = cv2.approxPolyDP(contour, 0.018 * perimeter, True)
        if len(approximation) == 4 and cv2.isContourConvex(approximation):
            candidates.append(_order_corners(approximation))
        rectangle = cv2.minAreaRect(contour)
        rectangle_area = max(1.0, rectangle[1][0] * rectangle[1][1])
        if cv2.contourArea(contour) / rectangle_area >= 0.58:
            candidates.append(_order_corners(cv2.boxPoints(rectangle)))
    return candidates


def _candidate_confidence(
    corners: np.ndarray, lightness: np.ndarray, source_shape: tuple[int, int]
) -> float:
    """Score whether a quadrilateral looks like one illuminated prescription page."""
    height, width = source_shape
    area_ratio = cv2.contourArea(corners.astype(np.float32)) / max(1.0, width * height)
    if area_ratio < 0.18 or area_ratio > 1.03:
        return 0.0
    contour = corners.astype(np.int32)
    interior = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(interior, contour, 255)
    interior_values = lightness[interior > 0]
    if not len(interior_values):
        return 0.0
    margin = cv2.dilate(interior, np.ones((11, 11), np.uint8))
    exterior_values = lightness[(margin > 0) & (interior == 0)]
    interior_lightness = float(np.median(interior_values))
    exterior_lightness = float(np.median(exterior_values)) if len(exterior_values) else 0.0
    # A page should occupy a substantial part of a phone image without blindly
    # accepting the entire camera frame. Very close scans remain valid.
    area_score = min(1.0, area_ratio / 0.58)
    separation_score = min(1.0, max(0.0, interior_lightness - exterior_lightness) / 35.0)
    edge_margin = np.min(
        np.concatenate((corners[:, 0], corners[:, 1], width - corners[:, 0], height - corners[:, 1]))
    )
    frame_score = 0.60 if edge_margin <= 3 else 1.0
    confidence = min(1.0, 0.45 * area_score + 0.40 * separation_score + 0.15 * frame_score)
    span_x = (corners[:, 0].max() - corners[:, 0].min()) / max(1, width)
    span_y = (corners[:, 1].max() - corners[:, 1].min()) / max(1, height)
    # A detected full-width light region that stops well above the bottom of a
    # phone photo is usually the prescription body, not the page boundary.
    if (span_x > 0.96 and span_y < 0.91) or (span_y > 0.96 and span_x < 0.91):
        confidence = min(confidence, 0.58)
    return confidence


def _detect_page(img: np.ndarray) -> Optional[PageDetection]:
    """Find a page through complementary paper, contour, and edge strategies."""
    height, width = img.shape[:2]
    scale = min(1.0, 1600 / max(height, width))
    preview = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    preview_height, preview_width = preview.shape[:2]
    lab = cv2.cvtColor(preview, cv2.COLOR_BGR2LAB)
    lightness = lab[:, :, 0]
    saturation = cv2.cvtColor(preview, cv2.COLOR_BGR2HSV)[:, :, 1]
    kernel_size = max(7, round(max(preview.shape[:2]) * 0.012))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))

    thresholds = {int(cv2.threshold(lightness, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[0]), 130, 150, 170, 190}
    candidates: list[tuple[np.ndarray, str]] = []
    for threshold in sorted(thresholds):
        paper = ((lightness >= threshold) & (saturation <= 175)).astype(np.uint8) * 255
        paper = cv2.morphologyEx(paper, cv2.MORPH_CLOSE, kernel, iterations=2)
        paper = cv2.morphologyEx(paper, cv2.MORPH_OPEN, kernel, iterations=1)
        candidates.extend((quad, f"paper-{threshold}") for quad in _contour_quad_candidates(paper))

    edges = cv2.Canny(cv2.GaussianBlur(lightness, (5, 5), 0), 35, 120)
    edges = cv2.dilate(edges, kernel, iterations=1)
    candidates.extend((quad, "page-edges") for quad in _contour_quad_candidates(edges))

    best: PageDetection | None = None
    for corners, method in candidates:
        confidence = _candidate_confidence(corners, lightness, (preview_height, preview_width))
        if best is None or confidence > best.confidence:
            best = PageDetection(corners / scale, confidence, method)
    return best


def _detect_page_corners(img: np.ndarray) -> Optional[np.ndarray]:
    """Backward-compatible page-corner accessor used by existing callers."""
    detection = _detect_page(img)
    return None if detection is None else detection.corners


def _warp_page(img: np.ndarray, corners: np.ndarray) -> np.ndarray:
    top_left, top_right, bottom_right, bottom_left = corners
    width = round(
        max(np.linalg.norm(top_right - top_left), np.linalg.norm(bottom_right - bottom_left))
    )
    height = round(
        max(np.linalg.norm(bottom_left - top_left), np.linalg.norm(bottom_right - top_right))
    )
    width, height = max(32, width), max(32, height)
    destination = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(corners.astype(np.float32), destination)
    return cv2.warpPerspective(
        img,
        transform,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _normalize_illumination(img: np.ndarray) -> np.ndarray:
    """Flatten camera shadows while retaining printed colors and watermarks."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lightness, channel_a, channel_b = cv2.split(lab)
    sigma = max(15.0, max(img.shape[:2]) / 28.0)
    background = cv2.GaussianBlur(lightness, (0, 0), sigmaX=sigma, sigmaY=sigma)
    normalized = cv2.divide(lightness, np.maximum(background, 1), scale=242)
    normalized = np.clip(normalized, 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.merge((normalized, channel_a, channel_b)), cv2.COLOR_LAB2BGR)


def _restore_page(img: np.ndarray) -> Image.Image:
    normalized = _normalize_illumination(img)
    denoised = cv2.bilateralFilter(normalized, d=5, sigmaColor=18, sigmaSpace=18)
    rgb = cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB)
    restored = Image.fromarray(rgb)
    restored = ImageEnhance.Contrast(restored).enhance(1.04)
    restored = restored.filter(ImageFilter.UnsharpMask(radius=1.2, percent=105, threshold=3))

    longest = max(restored.size)
    if longest < PRINT_LONG_EDGE_PX:
        scale = PRINT_LONG_EDGE_PX / longest
        restored = restored.resize(
            (round(restored.width * scale), round(restored.height * scale)),
            Image.Resampling.LANCZOS,
        )
        restored = restored.filter(ImageFilter.UnsharpMask(radius=0.8, percent=70, threshold=2))
    return restored


def _limit_resolution(
    img: np.ndarray, max_dimension: int = MAX_ANALYSIS_DIMENSION
) -> np.ndarray:
    height, width = img.shape[:2]
    longest_side = max(height, width)
    if longest_side <= max_dimension:
        return img
    scale = max_dimension / longest_side
    return cv2.resize(
        img,
        (round(width * scale), round(height * scale)),
        interpolation=cv2.INTER_AREA,
    )


def _manual_crop_corners(
    source: np.ndarray, crop: tuple[float, float, float, float]
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a user-confirmed normalized crop when auto geometry is uncertain."""
    height, width = source.shape[:2]
    left, top, right, bottom = crop
    left = max(0, min(width - 2, round(left * width)))
    top = max(0, min(height - 2, round(top * height)))
    right = max(left + 2, min(width, round(right * width)))
    bottom = max(top + 2, min(height, round(bottom * height)))
    cropped = source[top:bottom, left:right].copy()
    corners = np.array(
        [[left, top], [right - 1, top], [right - 1, bottom - 1], [left, bottom - 1]],
        dtype=np.float32,
    )
    return cropped, corners


def preprocess_document(
    image_bytes: bytes, manual_crop: Optional[tuple[float, float, float, float]] = None
) -> PreprocessResult:
    """Rectify a photographed page and create analysis and print masters."""
    source, source_size = _decode(image_bytes)
    if manual_crop is not None:
        canonical_cv, corners = _manual_crop_corners(source, manual_crop)
        confidence, method = 1.0, "manual-crop"
    else:
        detection = _detect_page(source)
        corners = None if detection is None else detection.corners
        confidence = 0.0 if detection is None else detection.confidence
        method = "full-image" if detection is None else detection.method
        canonical_cv = _warp_page(source, corners) if corners is not None else source.copy()
    canonical_rgb = cv2.cvtColor(canonical_cv, cv2.COLOR_BGR2RGB)
    canonical = Image.fromarray(canonical_rgb)
    restored = _restore_page(canonical_cv)
    analysis_cv = _limit_resolution(canonical_cv)
    analysis_cv = cv2.bilateralFilter(analysis_cv, d=5, sigmaColor=20, sigmaSpace=20)
    analysis = Image.fromarray(cv2.cvtColor(analysis_cv, cv2.COLOR_BGR2RGB))
    corner_tuple = None
    if corners is not None:
        corner_tuple = tuple((float(x), float(y)) for x, y in corners)
    return PreprocessResult(
        canonical=canonical,
        restored=restored,
        analysis=analysis,
        source_size=source_size,
        page_corners=corner_tuple,
        page_confidence=confidence,
        page_detection_method=method,
    )


def preprocess_image(image_bytes: bytes) -> Image.Image:
    """Backward-compatible analysis image used by the original pipeline."""
    return preprocess_document(image_bytes).analysis


def pil_to_bytes(img: Image.Image, fmt: str = "PNG") -> bytes:
    buffer = io.BytesIO()
    save_kwargs = {"optimize": True} if fmt.upper() == "PNG" else {}
    img.save(buffer, format=fmt, **save_kwargs)
    return buffer.getvalue()
