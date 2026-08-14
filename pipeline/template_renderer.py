"""Deterministic HTML renderers for fidelity and editable reconstruction."""

from __future__ import annotations

import base64
from html import escape
import io
import re

from PIL import Image

from pipeline.document_manifest import DocumentElement, DocumentManifest
from pipeline.page_geometry import PageGeometry


FONT_STACK = (
    "'Nirmala UI', 'Noto Sans Devanagari', 'Mangal', Arial, Helvetica, sans-serif"
)
LATIN_FONT_STACK = "Arial, Helvetica, sans-serif"


def _font_stack_for(element: DocumentElement) -> str:
    """Choose predictable local fonts instead of treating `sans-serif` as a font name."""
    requested = element.font_family.strip().lower()
    has_devanagari = any("\u0900" <= character <= "\u097f" for character in element.text)
    if has_devanagari:
        return FONT_STACK
    if requested in {"", "sans", "sans-serif", "arial", "helvetica", "inter", "roboto"}:
        return LATIN_FONT_STACK
    return f"'{escape(element.font_family, quote=True)}',{LATIN_FONT_STACK}"


def _image_data_uri(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def build_fidelity_html(image: Image.Image, page: PageGeometry) -> str:
    """Create a print document whose artwork is the restored source page."""
    data_uri = _image_data_uri(image)
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Faithful restored prescription</title>
<style>
@page {{ size: {page.css_size}; margin: 0; }}
* {{ box-sizing: border-box; }}
html, body {{ width: {page.width_mm:.3f}mm; height: {page.height_mm:.3f}mm; margin: 0; padding: 0; background: white; }}
.page {{ position: relative; width: 100%; height: 100%; overflow: hidden; background: white; }}
.page-image {{ display: block; width: 100%; height: 100%; object-fit: fill; }}
</style></head><body><div class="page"><img class="page-image" src="{data_uri}" alt="Restored prescription template"></div></body></html>"""


def inject_source_graphics(
    text_html: str,
    graphics: DocumentManifest,
    assets: dict[str, str],
    page: PageGeometry,
) -> str:
    """Overlay restored source artwork on top of an editable text layout.

    Gemini's HTML path remains responsible for real editable text. This keeps
    logo and watermark identity exact without making the model redraw them.
    """
    watermark_overlays: list[str] = []
    artwork_overlays: list[str] = []
    for element in graphics.elements:
        source = assets.get(element.id)
        if not source:
            continue
        box = element.box
        is_watermark = "watermark" in element.role.lower()
        # Watermarks are transparent source-ink assets. They must be above an
        # opaque generated page background, yet below identity artwork.
        z_index = 10 if is_watermark else max(30, element.z_index)
        opacity = element.opacity
        overlay = (
            '<img class="source-graphic-overlay '
            + ("source-watermark" if is_watermark else "source-artwork")
            + f'" src="{source}" alt="" style="'
            + f"left:{box.x * 100:.5f}%;top:{box.y * 100:.5f}%;"
            + f"width:{box.width * 100:.5f}%;height:{box.height * 100:.5f}%;"
            + f"opacity:{opacity:.4f};z-index:{z_index};\">"
        )
        (watermark_overlays if is_watermark else artwork_overlays).append(overlay)

    if not watermark_overlays and not artwork_overlays:
        return text_html

    overlay_style = """
<style id="source-graphics-contract">
body > .page, body > #prescription-page { position: relative !important; isolation: isolate; }
body > .page > :not(.source-graphics-layer), body > #prescription-page > :not(.source-graphics-layer) { position:relative; z-index:20; }
.source-graphics-layer { position:absolute !important; inset:0 !important; pointer-events:none !important; }
.source-watermark-layer { z-index:10 !important; }
.source-artwork-layer { z-index:30 !important; }
.source-graphic-overlay { position:absolute !important; display:block !important; object-fit:contain !important; pointer-events:none !important; }
.source-watermark { mix-blend-mode:multiply; }
svg.header-logo, svg.watermark-bg { display:none !important; }
</style>
"""
    head_close = re.search(r"</head\s*>", text_html, flags=re.IGNORECASE)
    if head_close is None:
        raise ValueError("Text reconstruction is not a complete HTML document.")
    with_style = text_html[: head_close.start()] + overlay_style + text_html[head_close.start() :]

    page_open = re.search(
        r'<div\b[^>]*(?:class=["\'][^"\']*\bpage\b[^"\']*["\']|'
        r'id=["\']prescription-page["\'])[^>]*>',
        with_style,
        flags=re.IGNORECASE,
    )
    if page_open is None:
        raise ValueError("Text reconstruction must contain a top-level page container.")

    layers: list[str] = []
    if watermark_overlays:
        layers.append(
            '<div class="source-graphics-layer source-watermark-layer" aria-hidden="true">'
            + "\n".join(watermark_overlays)
            + "</div>"
        )
    if artwork_overlays:
        layers.append(
            '<div class="source-graphics-layer source-artwork-layer" aria-hidden="true">'
            + "\n".join(artwork_overlays)
            + "</div>"
        )
    insertion = "\n".join(layers)
    return with_style[: page_open.end()] + insertion + with_style[page_open.end() :]


def _position_style(element: DocumentElement) -> str:
    box = element.box
    return (
        f"left:{box.x * 100:.5f}%;top:{box.y * 100:.5f}%;"
        f"width:{box.width * 100:.5f}%;height:{box.height * 100:.5f}%;"
        f"opacity:{element.opacity:.4f};z-index:{element.z_index};"
        f"transform:rotate({element.rotation:.3f}deg);"
    )


def _render_element(
    element: DocumentElement, assets: dict[str, str], page: PageGeometry
) -> str:
    common = _position_style(element)
    element_id = escape(element.id, quote=True)
    role = escape(element.role, quote=True)
    if element.kind == "image":
        source = assets.get(element.id)
        if not source:
            return ""
        return (
            f'<img id="{element_id}" class="document-element source-asset" '
            f'data-role="{role}" src="{source}" alt="" style="{common}object-fit:contain;">'
        )

    if element.kind == "line":
        thickness_mm = max(0.15, element.border_width * page.width_mm)
        vertical = element.box.height > element.box.width * 4
        line_style = (
            f"background:{element.color};"
            + (
                f"width:{thickness_mm:.3f}mm;"
                if vertical
                else f"height:{thickness_mm:.3f}mm;"
            )
        )
        return (
            f'<div id="{element_id}" class="document-element line-element" '
            f'data-role="{role}" style="{common}{line_style}"></div>'
        )

    border = ""
    if element.border_width > 0:
        border = (
            f"border:{element.border_width * page.width_mm:.4f}mm "
            f"solid {element.border_color};"
        )
    if element.kind == "shape":
        return (
            f'<div id="{element_id}" class="document-element shape-element" '
            f'data-role="{role}" style="{common}background:{element.background};{border}"></div>'
        )

    text = escape(element.text).replace("\n", "<br>")
    family = _font_stack_for(element)
    text_style = (
        f"font-family:{family};"
        f"font-size:{element.font_size * page.height_mm:.5f}mm;"
        f"font-weight:{element.font_weight};line-height:{element.line_height};"
        f"color:{element.color};background:{element.background};"
        f"text-align:{element.align};{border}"
    )
    return (
        f'<div id="{element_id}" class="document-element text-element" '
        f'data-role="{role}" data-fit-text="true" '
        f'data-align="{element.align}" style="{common}{text_style}">{text}</div>'
    )


def render_manifest_html(
    manifest: DocumentManifest, assets: dict[str, str], page: PageGeometry
) -> str:
    elements = "\n".join(
        _render_element(element, assets, page) for element in manifest.elements
    )
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Editable reconstructed prescription</title>
<style>
@page {{ size: {page.css_size}; margin: 0; }}
* {{ box-sizing: border-box; }}
html, body {{ width:{page.width_mm:.3f}mm; height:{page.height_mm:.3f}mm; margin:0; padding:0; background:{manifest.background}; }}
.page {{ position:relative; width:100%; height:100%; overflow:hidden; background:{manifest.background}; }}
.document-element {{ position:absolute; margin:0; padding:0; transform-origin:center center; }}
.text-element {{ overflow:visible; white-space:nowrap; letter-spacing:0; font-synthesis:none; }}
.source-asset {{ display:block; object-fit:contain; }}
</style>
<script id="manifest-text-fit">
(() => {{
  const fit = () => {{
    document.querySelectorAll('[data-fit-text="true"]').forEach((element) => {{
      const computed = window.getComputedStyle(element);
      const originalTransform = element.style.transform || "none";
      const align = element.dataset.align || "left";
      element.style.transformOrigin = `${{align === "right" ? "right" : align === "center" ? "center" : "left"}} top`;
      let size = parseFloat(computed.fontSize) || 10;
      const minSize = Math.max(3.5, size * 0.38);
      let attempts = 0;
      const overflows = () => element.scrollWidth > element.clientWidth + 0.5
        || element.scrollHeight > element.clientHeight + 0.5;
      while (overflows() && size > minSize && attempts < 48) {{
        size *= 0.94;
        element.style.fontSize = `${{size}}px`;
        attempts += 1;
      }}
      if (overflows()) {{
        const scale = Math.min(
          1,
          Math.max(0.2, (element.clientWidth - 0.5) / Math.max(1, element.scrollWidth)),
          Math.max(0.2, (element.clientHeight - 0.5) / Math.max(1, element.scrollHeight))
        );
        element.style.transform = `${{originalTransform}} scale(${{scale}})`;
      }}
    }});
    window.__prescriptionTextFitReady = true;
  }};
  if (document.fonts && document.fonts.ready) {{
    document.fonts.ready.then(fit);
  }} else {{
    fit();
  }}
}})();
</script></head><body><div class="page">
{elements}
</div></body></html>"""
