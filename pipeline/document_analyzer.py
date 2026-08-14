"""Multimodal analysis that returns structure instead of model-authored HTML."""

from __future__ import annotations

import time

from google import genai
from google.genai import types

from config import GEMINI_MODEL, get_gemini_api_key, use_openrouter
from pipeline.document_manifest import DocumentManifest
from pipeline.openrouter import generate_openrouter_content
from pipeline.page_geometry import PageGeometry


# A measured JSON manifest is far smaller than model-authored HTML/CSS. This
# accommodates dense multilingual pads while bounding the only AI request.
MANIFEST_OUTPUT_TOKENS = 16_384


ANALYSIS_PROMPT = """You are measuring a printed medical prescription template.
Return one JSON object only. Do not return HTML, markdown, or explanations.

The JSON format is:
{
  "background": "#rrggbb",
  "elements": [
    {
      "id": "stable-id",
      "kind": "text|image|line|shape",
      "role": "title|doctor|field|service|footer|logo|medical_icon|watermark_photo|watermark_seal|photo|seal|signature|other",
      "bbox": [x, y, width, height],
      "text": "exact visible text",
      "font_family": "closest font name",
      "font_size": 0.015,
      "font_weight": 400,
      "line_height": 1.15,
      "color": "#rrggbb",
      "background": "transparent",
      "opacity": 1.0,
      "align": "left|center|right",
      "border_color": "#rrggbb",
      "border_width": 0.0,
      "rotation": 0.0,
      "z_index": 1
    }
  ]
}

All bbox values and font_size are fractions of the complete page, from 0 to 1.
Measure the actual image; do not assume a standard prescription template.
This is a measurement task, not a design task: never invent, simplify, move, or
omit visible layout content. Transcribe every Hindi and English line exactly.
Keep separate lines, bullets, labels, and doctor details as separate text
elements so each one remains editable. Represent circular seals, photographs,
logos, signatures, medical icons, and watermarks as image elements so original
source pixels can be restored. Use `watermark_photo` for a faint portrait or
baby image and `watermark_seal` for a faint circular stamp. Do not describe or
redraw those graphics. Represent every divider and border as a line or shape.
Record the true opacity of watermarks. Preserve reading and visual z-order.
For every text element, make `bbox` cover the entire printed line at its stated
font_size. Never use a guessed or shortened transcription: if a line cannot be
read from the source, retain only the clearly visible characters and use the
tight source bounds rather than inventing words.
"""


GRAPHICS_PROMPT = """You are locating only non-text visual artwork in a printed
prescription template. Return one JSON object only. Do not return HTML,
markdown, explanations, text elements, divider lines, borders, rectangles, or
text boxes.

The JSON format is:
{
  "background": "#ffffff",
  "elements": [
    {
      "id": "stable-id",
      "kind": "image",
      "role": "logo|medical_icon|watermark_photo|watermark_seal|photo|seal|signature|other",
      "bbox": [x, y, width, height],
      "opacity": 1.0,
      "z_index": 1
    }
  ]
}

All bbox values are normalized from 0 to 1 relative to the complete page.
Identify only artwork that must remain source-derived: seals, logos, medical
icons, photos, watermarks, signatures, illustrations, and decorative emblems.
Classify a faint baby or portrait background image as `watermark_photo` and a
circular background stamp as `watermark_seal`; never use either role for
printed header text, a photo of the page, or a pale camera shadow. Use plain
`watermark` only when the artwork is neither a portrait nor a seal. Classify a
caduceus or other medical symbol as `medical_icon`. Include each
unique graphic once and return a box that contains the *entire* graphic. Never
allow a bbox edge to pass through a logo, circular ring, seal, photograph, or
watermark. Include a small paper margin when it is needed to avoid clipping;
including a little blank background is always preferable to cutting off artwork.
For a circular watermark, find its complete outer ring, even if it is faint. A
bbox must reach a page edge only when visible artwork genuinely reaches that
edge. Do not include any printed letters, numbers, bullet lists, or
horizontal/vertical rules as images.
"""


def analyze_document(image_bytes: bytes, page: PageGeometry) -> DocumentManifest:
    if use_openrouter():
        raw = generate_openrouter_content(
            image_bytes,
            "image/png",
            "You return only valid JSON for document measurement.",
            ANALYSIS_PROMPT + f"\nThe page is {page.css_size} ({page.orientation}).",
            MANIFEST_OUTPUT_TOKENS,
            {"type": "json_object"},
        )
        return DocumentManifest.from_json(raw)
    client = genai.Client(api_key=get_gemini_api_key())
    last_error: Exception | None = None
    for attempt in range(2):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=[
                    types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
                    ANALYSIS_PROMPT
                    + f"\nThe page is {page.css_size} ({page.orientation}).",
                ],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    max_output_tokens=MANIFEST_OUTPUT_TOKENS,
                    thinking_config=types.ThinkingConfig(thinking_level="high"),
                    http_options=types.HttpOptions(timeout=180_000),
                ),
            )
            return DocumentManifest.from_json(getattr(response, "text", "") or "")
        except Exception as exc:
            last_error = exc
            if attempt == 0:
                time.sleep(2)
    raise RuntimeError(f"Structured document analysis failed: {last_error}") from last_error


def analyze_graphics(image_bytes: bytes, page: PageGeometry) -> DocumentManifest:
    """Locate source artwork without asking the model to estimate text geometry."""
    if use_openrouter():
        raw = generate_openrouter_content(
            image_bytes,
            "image/png",
            "You return only valid JSON for graphic localization.",
            GRAPHICS_PROMPT + f"\nThe page is {page.css_size} ({page.orientation}).",
            6_144,
            {"type": "json_object"},
        )
        manifest = DocumentManifest.from_json(raw)
        graphics = tuple(element for element in manifest.elements if element.kind == "image")
        if not graphics:
            raise RuntimeError("Graphic analysis did not locate any source artwork.")
        return DocumentManifest(background=manifest.background, elements=graphics)
    client = genai.Client(api_key=get_gemini_api_key())
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
            GRAPHICS_PROMPT + f"\nThe page is {page.css_size} ({page.orientation}).",
        ],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            max_output_tokens=6_144,
            thinking_config=types.ThinkingConfig(thinking_level="high"),
            http_options=types.HttpOptions(timeout=120_000),
        ),
    )
    manifest = DocumentManifest.from_json(getattr(response, "text", "") or "")
    graphics = tuple(element for element in manifest.elements if element.kind == "image")
    if not graphics:
        raise RuntimeError("Graphic analysis did not locate any source artwork.")
    return DocumentManifest(background=manifest.background, elements=graphics)
