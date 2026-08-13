"""A bounded visual correction pass for structured document manifests."""

from __future__ import annotations

from dataclasses import asdict
import json

from google import genai
from google.genai import types

from config import GEMINI_MODEL, get_gemini_api_key
from pipeline.document_manifest import DocumentManifest
from pipeline.page_geometry import PageGeometry
from pipeline.visual_validation import VisualScore


REFINEMENT_PROMPT = """Compare the source prescription with the rendered reconstruction.
Return a corrected full manifest as one JSON object only. Preserve every exact text
value unless the source clearly proves it is wrong. Correct bounding boxes, font
sizes, line heights, colors, opacity, and image regions. Give special attention to
header height, column widths, patient rows, dividers, logo, watermark, and footer.
Do not redraw image elements; adjust their source crop boxes and placement. All
coordinates remain normalized from 0 to 1. Do not return HTML or markdown.
"""


def refine_manifest(
    source_png: bytes,
    rendered_png: bytes,
    manifest: DocumentManifest,
    page: PageGeometry,
    score: VisualScore,
) -> DocumentManifest:
    client = genai.Client(api_key=get_gemini_api_key())
    payload = json.dumps(asdict(manifest), ensure_ascii=False, separators=(",", ":"))
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[
            "SOURCE IMAGE:",
            types.Part.from_bytes(data=source_png, mime_type="image/png"),
            "CURRENT RENDER:",
            types.Part.from_bytes(data=rendered_png, mime_type="image/png"),
            REFINEMENT_PROMPT
            + f"\nPage: {page.css_size}. Current scores: {score.as_dict()}."
            + "\nCurrent manifest:\n"
            + payload,
        ],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            max_output_tokens=24_576,
            thinking_config=types.ThinkingConfig(thinking_level="high"),
            http_options=types.HttpOptions(timeout=180_000),
        ),
    )
    return DocumentManifest.from_json(getattr(response, "text", "") or "")

