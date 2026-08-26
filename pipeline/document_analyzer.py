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
       "role": "logo|medical_icon|watermark_photo|watermark_logo|watermark_seal|photo|seal|signature|other",
      "bbox": [x, y, width, height],
      "opacity": 1.0,
      "z_index": 1
    }
  ]
}

All bbox values are normalized from 0 to 1 relative to the complete page.
STRICT LOGO PRESENCE & DETECTION RULE:
• Check if a genuine clinic logo, emblem, hospital crest, or graphic illustration is visibly present in the top section of the page (y < 0.25).
• IF NO LOGO IS VISIBLY PRESENT IN THE SOURCE DOCUMENT, RETURN AN EMPTY ARRAY FOR LOGOS. Never invent, guess, or report a logo when none exists.
• IF A LOGO IS PRESENT, locate its exact tight bounding box enclosing ONLY the graphical emblem, excluding all text, doctor titles, addresses, and phone numbers.

WATERMARK DETECTION RULE:
• Carefully check the writable body area (y >= 0.18) for faint background watermarks.
• Classify a faint baby, child, or portrait photo in the body as `watermark_photo`.
• Classify a faint background hospital logo, emblem, or crest in the body as `watermark_logo`.
• Classify a circular background stamp as `watermark_seal`.
• CRITICAL: ONLY output a watermark box if a distinct, unmistakable faint background image, logo, or photo actually exists. DO NOT hallucinate a watermark.
• NEVER classify printed text, handwritten notes, dark pen strokes, empty space, or page shadows as a watermark.
CRITICAL CLASSIFICATION RULE: Any emblem, hospital crest, clinic logo, or
illustration located in the top header section (y < 0.25) MUST be classified
as `logo` or `medical_icon`, NEVER as `watermark_photo` or `watermark_seal`,
even if it depicts a baby, child, or stethoscope. Use `watermark_photo`,
`watermark_logo`, and `watermark_seal` ONLY for broad, faint background artwork
located in the main body of the prescription (y >= 0.18). Classify a caduceus
or other medical symbol as `medical_icon`.
TIGHT BOUNDING RULE FOR LOGOS & ILLUSTRATIONS: Never enclose printed clinic titles,
hospital names, phone numbers, addresses, or text blocks inside a logo bbox.
Bounding boxes for `logo` or `medical_icon` MUST tightly bound ONLY the non-text
graphical illustration itself (e.g. the small drawing, crest, or emblem), excluding
all surrounding text lines. Include each unique graphic once and return a box that
contains the *entire* non-text graphic.
TIGHT BOUNDING RULE FOR WATERMARKS & SEALS: Never enclose printed document text, patient info labels (Patient Name, Age, Sex, Wt, Date), phone numbers, or doctor titles inside a watermark bbox. Bounding boxes for `watermark_photo` (such as a faint baby image or portrait) MUST tightly bound ONLY the central artwork itself, excluding all surrounding text lines, labels, numbers, and headers. NEVER return a full-page bbox `[0, 0, 1, 1]` or a box covering the entire page width/height. Do not include any printed letters, numbers, bullet lists, or prescription body fields in an artwork bbox.
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
            analyzer_config = {
                "response_mime_type": "application/json",
                "max_output_tokens": MANIFEST_OUTPUT_TOKENS,
                "http_options": types.HttpOptions(timeout=180_000),
            }
            if GEMINI_MODEL.startswith("gemini-3"):
                analyzer_config["thinking_config"] = types.ThinkingConfig(thinking_level="high")
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=[
                    types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
                    ANALYSIS_PROMPT
                    + f"\nThe page is {page.css_size} ({page.orientation}).",
                ],
                config=types.GenerateContentConfig(**analyzer_config),
            )
            return DocumentManifest.from_json(getattr(response, "text", "") or "")
        except Exception as exc:
            last_error = exc
            if attempt == 0:
                time.sleep(2)
    raise RuntimeError(f"Structured document analysis failed: {last_error}") from last_error


def analyze_graphics(image_bytes: bytes, page: PageGeometry) -> DocumentManifest:
    """Locate source artwork without asking the model to estimate text geometry."""
    try:
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
        else:
            client = genai.Client(api_key=get_gemini_api_key())
            graphics_config = {
                "response_mime_type": "application/json",
                "max_output_tokens": 6_144,
                "http_options": types.HttpOptions(timeout=120_000),
            }
            if GEMINI_MODEL.startswith("gemini-3"):
                graphics_config["thinking_config"] = types.ThinkingConfig(thinking_level="high")
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=[
                    types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
                    GRAPHICS_PROMPT + f"\nThe page is {page.css_size} ({page.orientation}).",
                ],
                config=types.GenerateContentConfig(**graphics_config),
            )
            manifest = DocumentManifest.from_json(getattr(response, "text", "") or "")

        graphics = tuple(element for element in manifest.elements if element.kind == "image")
        return DocumentManifest(background=manifest.background, elements=graphics)
    except Exception:
        return DocumentManifest(background="#ffffff", elements=())
