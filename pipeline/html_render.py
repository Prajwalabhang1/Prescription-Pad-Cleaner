"""Render HTML to PNG and PDF for the prescription cleaner pipeline."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pymupdf  # PyMuPDF
from PIL import Image

from pipeline.page_geometry import PageGeometry


DEFAULT_PAGE = PageGeometry(210.0, 297.0)
PRINT_CONTRACT_MARKER = "prescription-print-contract"


class HtmlRenderError(RuntimeError):
    """Raised when generated HTML does not produce one printable page."""


def prepare_print_html(html: str, page: PageGeometry) -> str:
    """Apply the single-page print contract to model-generated HTML."""
    if PRINT_CONTRACT_MARKER in html:
        return html

    css = f"""
<style id="{PRINT_CONTRACT_MARKER}">
@page {{
  size: {page.css_size};
  margin: 0;
}}

html,
body {{
  width: {page.width_mm:.3f}mm !important;
  height: {page.height_mm:.3f}mm !important;
  min-width: {page.width_mm:.3f}mm !important;
  min-height: {page.height_mm:.3f}mm !important;
  max-width: {page.width_mm:.3f}mm !important;
  max-height: {page.height_mm:.3f}mm !important;
  margin: 0 !important;
  padding: 0 !important;
  background: #ffffff !important;
}}

body {{
  display: block !important;
  overflow: hidden !important;
}}

body > .page,
body > #prescription-page {{
  box-sizing: border-box !important;
  width: {page.width_mm:.3f}mm !important;
  height: {page.height_mm:.3f}mm !important;
  min-height: {page.height_mm:.3f}mm !important;
  max-height: {page.height_mm:.3f}mm !important;
  margin: 0 !important;
  box-shadow: none !important;
  break-before: avoid-page !important;
  break-after: avoid-page !important;
  break-inside: avoid-page !important;
  page-break-before: avoid !important;
  page-break-after: avoid !important;
  page-break-inside: avoid !important;
}}

body > .page .patient-info,
body > .page .patient-info .info-row {{
  max-width: 100% !important;
}}

body > .page .patient-info .field {{
  min-width: 0 !important;
  overflow: hidden !important;
}}
</style>
"""

    head_close = "</head>"
    if head_close not in html.lower():
        raise HtmlRenderError("Generated HTML does not contain a closing </head> tag.")

    start = html.lower().index(head_close)
    return html[:start] + css + html[start:]


def _render_with_weasyprint(html: str, resolution: int) -> tuple[Image.Image, bytes]:
    """Render with WeasyPrint when the native dependencies are available."""
    from weasyprint import HTML

    pdf_bytes = HTML(string=html).write_pdf()
    pdf_doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    if pdf_doc.page_count != 1:
        page_count = pdf_doc.page_count
        pdf_doc.close()
        raise HtmlRenderError(
            "Generated artwork overflowed its page and produced "
            f"{page_count} PDF pages; no output was created."
        )

    zoom = resolution / 72
    pix = pdf_doc[0].get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
    pil_image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    pdf_doc.close()
    return pil_image, pdf_bytes


def _render_with_playwright(
    html: str, resolution: int, page: PageGeometry
) -> tuple[Image.Image, bytes]:
    """Render with Chromium when WeasyPrint is not usable on this machine."""
    from playwright.sync_api import sync_playwright

    viewport_width = max(1, round(page.width_mm / 25.4 * resolution))
    viewport_height = max(1, round(page.height_mm / 25.4 * resolution))

    html_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", suffix=".html", delete=False, encoding="utf-8"
        ) as html_file:
            html_file.write(html)
            html_path = Path(html_file.name)

        with sync_playwright() as p:
            browser = p.chromium.launch()
            page_obj = browser.new_page(
                viewport={"width": viewport_width, "height": viewport_height},
                device_scale_factor=1,
            )
            page_obj.goto(html_path.as_uri(), wait_until="load")
            page_obj.evaluate("document.fonts.ready")
            page_obj.wait_for_function(
                "Array.from(document.images).every((image) => image.complete && image.naturalWidth > 0)"
            )
            page_obj.emulate_media(media="print")
            pdf_bytes = page_obj.pdf(
                print_background=True,
                prefer_css_page_size=True,
            )
            browser.close()

        pdf_doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        if pdf_doc.page_count != 1:
            page_count = pdf_doc.page_count
            pdf_doc.close()
            raise HtmlRenderError(
                "Generated artwork overflowed its page and produced "
                f"{page_count} PDF pages; no output was created."
            )

        zoom = resolution / 72
        pix = pdf_doc[0].get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
        pil_image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        pdf_doc.close()
        return pil_image, pdf_bytes
    finally:
        if html_path is not None:
            html_path.unlink(missing_ok=True)


def render_html(
    html: str,
    resolution: int = 300,
    page: PageGeometry = DEFAULT_PAGE,
) -> tuple[Image.Image, bytes]:
    """Render *html* to (PIL.Image, pdf_bytes)."""
    printable_html = prepare_print_html(html, page)

    # WeasyPrint's Pango/GObject stack is not bundled on Windows. Chromium is
    # deterministic here and avoids emitting a native-library error before the
    # fallback succeeds.
    if os.name == "nt":
        return _render_with_playwright(printable_html, resolution, page)

    try:
        return _render_with_weasyprint(printable_html, resolution)
    except Exception as weasy_error:
        try:
            return _render_with_playwright(printable_html, resolution, page)
        except Exception as playwright_error:
            raise HtmlRenderError(
                "Both available HTML renderers failed. "
                f"Chromium: {playwright_error}. WeasyPrint: {weasy_error}"
            ) from playwright_error
