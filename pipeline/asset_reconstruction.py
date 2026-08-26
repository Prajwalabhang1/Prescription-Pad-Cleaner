"""Restore source graphics for exact placement in reconstructed layouts."""

from __future__ import annotations

import base64
import io
from dataclasses import replace

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

from pipeline.document_manifest import DocumentManifest, NormalizedBox


def _role_family(role: str) -> str:
    normalized = role.lower().replace("-", "_").replace(" ", "_")
    if "watermark" in normalized:
        return "watermark"
    if "seal" in normalized:
        return "seal"
    if normalized in {"logo", "medical_icon", "icon", "emblem", "illustration"}:
        return "logo"
    if "signature" in normalized:
        return "signature"
    if "photo" in normalized:
        return "photo"
    return "other"


def _watermark_kind(role: str) -> str:
    """Return the artwork subtype without losing the model's semantic hint."""
    normalized = role.lower().replace("-", "_").replace(" ", "_")
    if "watermark" not in normalized:
        return ""
    if "photo" in normalized or "portrait" in normalized or "baby" in normalized:
        return "photo"
    if "seal" in normalized or "stamp" in normalized or "emblem" in normalized:
        return "seal"
    return "generic"


def _box_from_pixels(
    left: float, top: float, right: float, bottom: float, width: int, height: int
) -> NormalizedBox:
    """Convert a pixel rectangle to a page-bounded normalized box."""
    return NormalizedBox.from_value(
        [left / width, top / height, (right - left) / width, (bottom - top) / height]
    )


def _expanded_box(box: NormalizedBox, horizontal: float, vertical: float) -> NormalizedBox:
    return NormalizedBox.from_value(
        [
            box.x - horizontal,
            box.y - vertical,
            box.width + horizontal * 2,
            box.height + vertical * 2,
        ]
    )


def _small_rgb(source: Image.Image, target_width: int = 512) -> np.ndarray:
    """Use a bounded working image so the guard is fast on high-DPI pages."""
    rgb = source.convert("RGB")
    if rgb.width > target_width:
        height = max(1, round(rgb.height * target_width / rgb.width))
        rgb = rgb.resize((target_width, height), Image.Resampling.LANCZOS)
    return np.asarray(rgb)


def _intersection_over_union(first: NormalizedBox, second: NormalizedBox) -> float:
    left = max(first.x, second.x)
    top = max(first.y, second.y)
    right = min(first.x + first.width, second.x + second.width)
    bottom = min(first.y + first.height, second.y + second.height)
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    union = first.width * first.height + second.width * second.height - intersection
    return intersection / union if union else 0.0





def _find_logo_circle(rgb: np.ndarray, proposal: NormalizedBox) -> NormalizedBox | None:
    """Find a complete top-left circular emblem without including header text."""
    height, width = rgb.shape[:2]
    right_limit = max(1, min(width, round(width * 0.23)))
    bottom_limit = max(1, min(height, round(height * 0.17)))
    gray = cv2.cvtColor(rgb[:bottom_limit, :right_limit], cv2.COLOR_RGB2GRAY)
    circles = cv2.HoughCircles(
        cv2.GaussianBlur(gray, (5, 5), 0),
        cv2.HOUGH_GRADIENT,
        dp=1.1,
        minDist=max(18, round(width * 0.04)),
        param1=75,
        param2=19,
        minRadius=max(10, round(width * 0.035)),
        maxRadius=max(12, round(width * 0.15)),
    )
    if circles is None:
        return None
    target_x = (proposal.x + proposal.width / 2) * width
    target_y = (proposal.y + proposal.height / 2) * height
    best: tuple[float, NormalizedBox] | None = None
    for center_x, center_y, radius in np.round(circles[0]).astype(int):
        if center_x - radius <= 1 or center_y - radius <= 1:
            continue
        if center_x + radius >= right_limit - 1 or center_y + radius >= bottom_limit - 1:
            continue
        target_x = (proposal.x + proposal.width / 2) * width
        target_y = (proposal.y + proposal.height / 2) * height
        center_error = abs(center_x - target_x) / width + abs(center_y - target_y) / height
        if center_error > 0.09:
            continue
        candidate = _box_from_pixels(
            center_x - radius,
            center_y - radius,
            center_x + radius,
            center_y + radius,
            width,
            height,
        )
        if _intersection_over_union(candidate, proposal) < 0.15:
            continue
        radius_error = abs((radius * 2) / width - proposal.width)
        score = center_error * 2.0 + radius_error
        if best is None or score < best[0]:
            best = (score, candidate)
    if best is None:
        return None
    return _expanded_box(best[1], 0.006, 0.006)


def _find_watermark_circle(rgb: np.ndarray, proposal: NormalizedBox) -> NormalizedBox | None:
    """Recover a faint circular seal when a vision bbox cuts through its ring."""
    height, width = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(32, round(width * 0.15)),
        param1=70,
        param2=24,
        minRadius=max(12, round(width * 0.04)),
        maxRadius=max(14, round(width * 0.38)),
    )
    if circles is None:
        return None

    target_center_x = proposal.x + proposal.width / 2
    target_center_y = proposal.y + proposal.height / 2
    best: tuple[float, NormalizedBox] | None = None
    for center_x, center_y, radius in np.round(circles[0]).astype(int):
        candidate = _box_from_pixels(
            center_x - radius,
            center_y - radius,
            center_x + radius,
            center_y + radius,
            width,
            height,
        )
        if candidate.x + candidate.width < 0.15 or candidate.y + candidate.height < 0.28:
            continue
        width_error = abs(candidate.width - proposal.width)
        height_error = abs(candidate.height - proposal.height)
        center_error = abs(center_x / width - target_center_x) + abs(
            center_y / height - target_center_y
        )
        score = width_error * 3.0 + height_error * 3.0 + center_error * 0.35
        if best is None or score < best[0]:
            best = (score, candidate)
    if best is None:
        return None
    candidate = best[1]
    # Reject unrelated circles: we only trust a circle with a broadly matching
    # visual size to the artwork proposed by the vision model.
    if (
        abs(candidate.width - proposal.width) > 0.16
        or abs(candidate.height - proposal.height) > 0.13
    ):
        return None
    return _expanded_box(candidate, 0.012, 0.012)


def _is_near_square(box: NormalizedBox) -> bool:
    ratio = box.width / max(box.height, 0.0001)
    return 0.70 <= ratio <= 1.45


def sanitize_graphics(manifest: DocumentManifest) -> tuple[DocumentManifest, tuple[str, ...]]:
    """Drop model candidates that are structurally impossible source artwork."""
    accepted = []
    warnings: list[str] = []
    for element in manifest.elements:
        if element.kind != "image":
            accepted.append(element)
            continue
        family = _role_family(element.role)
        box = element.box
        area = box.width * box.height
        if family == "watermark":
            # Header artwork (y < 0.25) is a logo/emblem, not a background watermark.
            if box.y < 0.25 and box.width < 0.50:
                element = replace(element, role="logo", opacity=1.0)
                family = "logo"
            elif box.y < 0.16 or box.height < 0.12:
                warnings.append(f"Ignored {element.id}: watermark candidate overlaps the header or is too small.")
                continue
            elif box.y > 0.55:
                warnings.append(f"Ignored {element.id}: watermark candidate is in the bottom footer region.")
                continue
            elif box.width > 0.50 and (box.width / box.height > 2.2):
                warnings.append(f"Ignored {element.id}: watermark candidate is a wide rectangle ({box.width:.2f}x{box.height:.2f}), likely hallucinated text.")
                continue
            elif box.width > 0.88 or box.height > 0.78 or area > 0.58:
                warnings.append(f"Ignored {element.id}: watermark candidate is too broad ({box.width:.2f}x{box.height:.2f}) and encloses printed document text.")
                continue
        if family in {"logo", "seal"}:
            if box.y > 0.42 or area > 0.20:
                warnings.append(f"Ignored {element.id}: logo candidate has implausible page coverage.")
                continue
        accepted.append(element)
    return replace(manifest, elements=tuple(accepted)), tuple(warnings)


def complete_graphic_bounds(source: Image.Image, manifest: DocumentManifest) -> DocumentManifest:
    """Repair graphic boxes that would visibly crop source artwork.

    Gemini supplies the semantic role and an initial location. This local guard
    uses the original pixels to preserve the complete logo or circular seal when
    the proposed box ends within the artwork itself.
    """
    rgb = _small_rgb(source)
    corrected = []
    for element in manifest.elements:
        if element.kind != "image":
            corrected.append(element)
            continue
        role = _role_family(element.role)
        box = element.box
        if role == "seal" or (role == "logo" and _is_near_square(box)):
            box = _find_logo_circle(rgb, box) or (
                _expanded_box(box, 0.008, 0.008) if role == "seal" else box
            )
        elif role == "watermark":
            watermark_kind = _watermark_kind(element.role)
            box = (
                _find_watermark_circle(rgb, box)
                if watermark_kind != "photo" and _is_near_square(box)
                else _expanded_box(box, 0.012, 0.014)
            )
            if box is None:
                box = _expanded_box(element.box, 0.012, 0.012)
        else:
            box = _expanded_box(box, 0.004, 0.004)
        opacity = 1.0 if role == "watermark" else element.opacity
        corrected.append(replace(element, box=box, opacity=opacity))
    return replace(manifest, elements=tuple(corrected))



def _crop_box(image: Image.Image, box) -> tuple[int, int, int, int]:
    left = max(0, round(box.x * image.width))
    top = max(0, round(box.y * image.height))
    right = min(image.width, round((box.x + box.width) * image.width))
    bottom = min(image.height, round((box.y + box.height) * image.height))
    return left, top, max(left + 1, right), max(top + 1, bottom)


def _restore_asset(crop: Image.Image, role: str) -> Image.Image:
    rgb = np.asarray(crop.convert("RGB"))
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    lightness, channel_a, channel_b = cv2.split(lab)
    clip_limit = 1.15 if role == "watermark" else 1.7
    lightness = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(6, 6)).apply(lightness)
    restored = cv2.cvtColor(
        cv2.merge((lightness, channel_a, channel_b)), cv2.COLOR_LAB2RGB
    )
    output = Image.fromarray(restored)
    contrast = 1.01 if role == "watermark" else 1.08
    output = ImageEnhance.Contrast(output).enhance(contrast)
    output = output.filter(ImageFilter.UnsharpMask(radius=0.8, percent=75, threshold=3))
    if max(output.size) < 1200:
        scale = min(4.0, 1200 / max(output.size))
        output = output.resize(
            (round(output.width * scale), round(output.height * scale)),
            Image.Resampling.LANCZOS,
        )
    return output


def _clean_paper_background(crop: Image.Image, is_watermark: bool = False, max_opacity: float = 1.0) -> Image.Image:
    """Preprocess logo / watermark asset:
    1. White balance & remove local paper lighting/tint.
    2. Isolate logo ink / artwork lines.
    3. Enhance color contrast, saturation, and sharpness.
    """
    rgb = np.asarray(crop.convert("RGB"))
    height, width = rgb.shape[:2]

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    saturation = hsv[:, :, 1]

    # Estimate local paper background lightness and saturation
    blur_k = max(15, round(min(width, height) * 0.35))
    if blur_k % 2 == 0:
        blur_k += 1
    paper_bg_gray = cv2.GaussianBlur(gray, (blur_k, blur_k), 0)
    paper_bg_sat = cv2.GaussianBlur(saturation, (blur_k, blur_k), 0)

    # Difference relative to paper background
    ink_luma_diff = np.maximum(paper_bg_gray - gray, 0.0)
    ink_sat_diff = np.maximum(saturation - paper_bg_sat, 0.0)

    if is_watermark:
        # Faint line artwork / watermark in main body
        alpha_luma = np.clip((ink_luma_diff - 2.0) / 22.0, 0.0, max_opacity)
        alpha_sat = np.clip((ink_sat_diff - 3.0) / 25.0, 0.0, max_opacity)
        alpha = np.maximum(alpha_luma, alpha_sat)
    else:
        # Header Logo / Icon
        alpha_luma = np.clip((ink_luma_diff - 8.0) / 28.0, 0.0, 1.0)
        alpha_sat = np.clip((ink_sat_diff - 12.0) / 30.0, 0.0, 1.0)
        alpha = np.maximum(alpha_luma, alpha_sat)

    # Edge feathering
    feather = max(2, round(min(width, height) * 0.03))
    edge_mask = np.ones((height, width), dtype=np.float32)
    ramp = np.linspace(0.0, 1.0, feather, endpoint=True)
    edge_mask[:feather, :] *= ramp[:, None]
    edge_mask[-feather:, :] *= ramp[::-1, None]
    edge_mask[:, :feather] *= ramp[None, :]
    edge_mask[:, -feather:] *= ramp[None, ::-1]
    alpha = alpha * edge_mask
    alpha = cv2.GaussianBlur(alpha, (0, 0), sigmaX=0.45)

    # White balance against estimated paper background so paper tint becomes clean
    normalized_rgb = rgb.astype(np.float32)
    for c in range(3):
        bg_c = cv2.GaussianBlur(normalized_rgb[:, :, c], (blur_k, blur_k), 0)
        bg_c = np.maximum(bg_c, 1.0)
        normalized_rgb[:, :, c] = np.clip(normalized_rgb[:, :, c] * (255.0 / bg_c), 0.0, 255.0)

    clean_rgb = Image.fromarray(normalized_rgb.astype(np.uint8))
    clean_rgb = ImageEnhance.Color(clean_rgb).enhance(1.30)
    clean_rgb = ImageEnhance.Contrast(clean_rgb).enhance(1.20)
    clean_rgb = clean_rgb.filter(ImageFilter.UnsharpMask(radius=1.0, percent=85, threshold=2))

    rgba = clean_rgb.convert("RGBA")
    rgba.putalpha(Image.fromarray(np.round(alpha * 255).astype(np.uint8)))
    return rgba


def _transparent_logo(crop: Image.Image) -> Image.Image:
    return _clean_paper_background(crop, is_watermark=False, max_opacity=1.0)


def _transparent_watermark(crop: Image.Image, max_opacity: float = 0.38) -> Image.Image:
    return _clean_paper_background(crop, is_watermark=True, max_opacity=max_opacity)


def _transparent_photo_watermark(crop: Image.Image) -> Image.Image:
    return _clean_paper_background(crop, is_watermark=True, max_opacity=0.28)


def _crop_circular_photo(crop: Image.Image) -> Image.Image:
    """Keep the full RGB photo inside a clean circular mask, removing square crop corners."""
    rgb = crop.convert("RGB")
    width, height = rgb.size

    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((0, 0, width, height), fill=255)

    rgba = rgb.copy()
    rgba.putalpha(mask)
    return rgba


def _asset_with_alpha(crop: Image.Image, role: str) -> Image.Image:
    family = _role_family(role)
    if "photo" in role and "watermark" not in role:
        return _crop_circular_photo(crop)
    if family in {"logo", "seal", "signature"}:
        return _transparent_logo(crop)
    if family == "watermark":
        if _watermark_kind(role) == "photo":
            return _transparent_photo_watermark(crop)
        return _transparent_watermark(crop)
    return _restore_asset(crop, role)



def _data_uri(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def reconstruct_assets(
    source: Image.Image, manifest: DocumentManifest
) -> dict[str, str]:
    """Return restored embedded PNGs for every image element in the manifest."""
    assets: dict[str, str] = {}
    for element in manifest.elements:
        if element.kind != "image":
            continue
        crop = source.crop(_crop_box(source, element.box))
        assets[element.id] = _data_uri(_asset_with_alpha(crop, element.role.lower()))
    return assets
