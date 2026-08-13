from typing import Any, Dict, List, Optional

from PIL import Image, ImageDraw, ImageFont

from config import CANVAS_SIZE, TEMPLATE


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    path = TEMPLATE["font_bold"] if bold else TEMPLATE["font_regular"]
    try:
        return ImageFont.truetype(path, size)
    except Exception:
        return ImageFont.load_default()


def _crop_logo(source: Image.Image) -> Optional[Image.Image]:
    """Heuristic: clinic logos sit in the top-left corner of the scan."""
    w, h = source.size
    box = (0, 0, int(w * 0.28), int(h * 0.20))
    crop = source.crop(box)
    return crop if crop.size[0] > 10 and crop.size[1] > 10 else None


def build_clean_pad(source: Image.Image, manifest: Dict[str, Any]) -> Image.Image:
    """
    Draws a cleaned prescription pad from the OCR manifest onto a fresh A4
    canvas. This stands in for Canva create + autofill + export while no
    Canva Connect token is available.
    """
    W, H = CANVAS_SIZE
    canvas = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(canvas)
    margin = TEMPLATE["margin"]
    header_h = TEMPLATE["header_height"]
    footer_h = TEMPLATE["footer_height"]

    # Logo
    logo = _crop_logo(source)
    if logo:
        logo_w = 220
        logo_h = int(logo.size[1] * (logo_w / logo.size[0]))
        logo = logo.resize((logo_w, logo_h))
        canvas.paste(logo, (margin, margin))

    # Header text: clinic / doctor name + qualifications + reg no
    text_x = margin + 240
    text_y = margin
    header_lines: List[str] = [l["text"] for l in manifest.get("header_lines", [])]
    if header_lines:
        draw.text((text_x, text_y), header_lines[0], font=_font(30, bold=True), fill="black")
        y = text_y + 42
        for line in header_lines[1:4]:
            draw.text((text_x, y), line, font=_font(16), fill="#222222")
            y += 24
    else:
        draw.text((text_x, text_y), "Clinic Name", font=_font(30, bold=True), fill="black")

    draw.line([(margin, header_h), (W - margin, header_h)], fill="#999999", width=2)

    # Rx symbol + ruled lines for the blank body area
    body_top = header_h + 40
    body_bottom = H - footer_h - 20
    draw.text((margin, body_top - 10), "Rx", font=_font(34, bold=True), fill="#333333")
    for y in range(body_top + 50, body_bottom, 50):
        draw.line([(margin, y), (W - margin, y)], fill="#dddddd", width=1)

    # Footer: address / phone / reg no
    footer_lines: List[str] = [l["text"] for l in manifest.get("footer_lines", [])]
    draw.line([(margin, H - footer_h), (W - margin, H - footer_h)], fill="#999999", width=2)
    y = H - footer_h + 20
    if footer_lines:
        for line in footer_lines[:3]:
            draw.text((margin, y), line, font=_font(14), fill="#333333")
            y += 22
    else:
        draw.text((margin, y), "Address line  |  Phone  |  Reg. No.", font=_font(14), fill="#333333")

    return canvas
