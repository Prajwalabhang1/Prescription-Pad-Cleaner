"""
pipeline/pdf_export.py
──────────────────────
Standalone PDF export from a PIL Image (fallback if WeasyPrint is not
used directly).
"""

import io

from PIL import Image


def image_to_pdf_bytes(img: Image.Image) -> bytes:
    """Convert a PIL Image to PDF bytes (single page)."""
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PDF", resolution=300.0)
    return buf.getvalue()
