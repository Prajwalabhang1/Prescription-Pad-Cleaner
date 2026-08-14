"""
pipeline/gemini_vision.py
─────────────────────────
Sends the input prescription image to Gemini 3.6 Flash and gets back a
self-contained HTML/CSS reproduction that matches the clean reference.
"""

from html import unescape
import io
import re
import time
from typing import Any

from google import genai
from google.genai import types
from PIL import Image

from config import GEMINI_MODEL, get_gemini_api_key, use_openrouter
from pipeline.page_geometry import PageGeometry
from pipeline.openrouter import generate_openrouter_content

SYSTEM_PROMPT = """\
You are an expert document designer specialising in medical/clinical stationery.

You will be given an image of a scanned prescription pad (possibly low quality,
skewed, noisy, or photographed at an angle). Your task is to:

1. Analyse the page's text (Hindi Devanagari + English), rules, boxes,
   colours, and exact spatial layout.
2. Generate a **single, self-contained HTML file with inline CSS** that
   recreates the clean, editable text and layout. Every letter must be real
   HTML text, never a screenshot, canvas, SVG text, image, or base64 asset.

CRITICAL REQUIREMENTS:
 • Reproduce the exact layout you observe. Do not assume a standard
   prescription layout, logo position, caduceus, banner, or field list.
 • Preserve ALL text in BOTH Hindi (Devanagari) and English exactly as shown.
 • Never infer, complete, translate, normalize, or invent text from medical
   context. Copy only characters visibly present in the source; do not add
   clinic details, services, contact numbers, or labels that are not visible.
 • Match colours precisely — use the exact hex/rgb values you observe.
 • Use Google Fonts:
   – 'Noto Sans Devanagari' (weights 400, 700) for Hindi text.
   – A suitable serif font for stylised English names (e.g. 'Playfair Display'
     or 'EB Garamond').
   – A clean sans-serif for smaller English lines (e.g. 'Inter' or 'Roboto').
 • Do not invent or redraw logos, seals, photographs, signatures, or
   watermarks. Leave their regions transparent; another stage overlays the
   original source artwork at exactly those locations.
 • Ignore camera surroundings, desk/floor backgrounds, paper holes, wrinkles,
   glare, shadows, and handwritten pen marks. Do not recreate them in HTML.
 • Use CSS only for plain rules, boxes, fills, and borders.
 • The print page dimensions and orientation are supplied with the request.
   Use those exact dimensions with a white background; do not force A4.
 • Use exactly one top-level `<div class="page">` inside `<body>`. It must
   contain the entire prescription and have no external margin or screen-only
   padding. Do not add a second page, an outer preview wrapper, or a print
   margin around this element.
 • Keep every header column and patient-field row within the page width. In
   flex layouts, set `min-width: 0` on shrinkable columns and never let fixed
   field widths add up to more than the available row width.
 • Preserve only fields and symbols actually visible in the image.
 • Keep the body area faithful to the supplied page and do not add features.
 • Add @media print styles so the page prints cleanly at 100 % scale.
 • Add a thin border/outline around the entire page (1px solid #ccc).

OUTPUT FORMAT:
 • Return ONLY the complete HTML code.
 • Do NOT include markdown formatting, code fences, or explanations.
 • The HTML must start with <!DOCTYPE html> and end with </html>.
 • All CSS must be in a <style> tag inside <head>.
 • Use Noto Sans Devanagari for Hindi and Arial or Roboto for English where
   appropriate. Set explicit widths, heights, font sizes, and line-heights so
   text never wraps, clips, overflows, or shifts when rendered to PDF.
 • Keep the stylesheet concise (target under 8,000 output tokens) so the
   complete document finishes in one response. Never omit a closing </style>,
   </body>, or </html> tag.
"""

# The first pass is deliberately compact and fast. A larger second pass is
# reserved for the rare document that genuinely needs it.
MAX_GENERATION_ATTEMPTS = 3
PRIMARY_OUTPUT_TOKENS = 16_384
RETRY_OUTPUT_TOKENS = 24_576
# A 90-second client deadline is too close to the service-side deadline for a
# detailed image + long HTML response. Leave the service enough time to finish
# its final output tokens, while still bounding a request from the UI.
GENERATION_TIMEOUT_MS = 150_000
TRANSIENT_RETRY_DELAYS_SECONDS = (2, 5)
RETRY_IMAGE_MAX_DIMENSION = 1600
MAX_RATE_LIMIT_WAIT_SECONDS = 60

class IncompleteHtmlError(RuntimeError):
    """Raised when a model response is not a complete HTML document."""


class GeminiRateLimitError(RuntimeError):
    """Raised after Gemini's requested quota wait could not restore access."""


def _is_rate_limit_error(error: Exception) -> bool:
    message = str(error).upper()
    return "RESOURCE_EXHAUSTED" in message or " 429" in message


def _retry_delay_seconds(error: Exception) -> int | None:
    """Read Gemini's RetryInfo delay without depending on SDK error classes."""
    match = re.search(
        r"retryDelay.*?(\d+(?:\.\d+)?)s", str(error), re.IGNORECASE | re.DOTALL
    )
    if not match:
        return None
    return max(1, round(float(match.group(1))))


def _is_transient_api_error(error: Exception) -> bool:
    """Return whether Gemini's failure is worth retrying without user action."""
    message = str(error).upper()
    return any(
        marker in message
        for marker in (
            "DEADLINE_EXCEEDED",
            "TIMED OUT",
            "TIMEOUT",
            "UNAVAILABLE",
            "INTERNAL",
            " 500",
            " 502",
            " 503",
            " 504",
        )
    )


def _compact_retry_image(image_bytes: bytes) -> bytes:
    """Make a smaller, high-quality retry payload after a service timeout.

    The first attempt retains the exact preprocessed PNG. Only a failed
    transient request switches to JPEG, which materially reduces upload and
    vision latency while retaining enough detail for document layout recovery.
    """
    try:
        with Image.open(io.BytesIO(image_bytes)) as source:
            image = source.convert("RGB")
            longest_edge = max(image.size)
            if longest_edge > RETRY_IMAGE_MAX_DIMENSION:
                scale = RETRY_IMAGE_MAX_DIMENSION / longest_edge
                image = image.resize(
                    (round(image.width * scale), round(image.height * scale)),
                    Image.Resampling.LANCZOS,
                )
            output = io.BytesIO()
            image.save(output, format="JPEG", quality=92, optimize=True)
            return output.getvalue()
    except Exception:
        # If a non-image mock or damaged input reaches this branch, retain the
        # original request payload and let Gemini report the real issue.
        return image_bytes


def _extract_html(text: str) -> str:
    """Extract an HTML document while preserving incomplete output for validation."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:html)?\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
    text = text.strip()

    # Models occasionally prepend a short explanation despite the prompt. Keep
    # only the document itself when it contains a clear opening tag.
    start = re.search(r"<!doctype\s+html\b|<html\b", text, flags=re.IGNORECASE)
    if start:
        text = text[start.start() :]

    # Drop trailing prose only after a complete document. An absent closing
    # ``</html>`` must be left intact so validation can reject it.
    end = re.search(r"</html\s*>", text, flags=re.IGNORECASE)
    if end:
        text = text[: end.end()]

    return text.strip()


def validate_reconstruction_html(html: str) -> str:
    """Return *html* only when it is a complete document safe to render.

    WeasyPrint is intentionally forgiving: it will render a page even when a
    response ends halfway through a stylesheet. That produces a silent blank
    prescription. Require the document boundaries that the generation prompt
    promises before handing its output to the renderer.
    """
    if not html or not isinstance(html, str):
        raise IncompleteHtmlError("Gemini returned no HTML content.")

    checks = (
        (r"^\s*(?:<!doctype\s+html\b|<html\b)", "an HTML opening tag"),
        (r"<head\b[^>]*>", "a <head> section"),
        (r"</head\s*>", "a closing </head> tag"),
        (r"<body\b[^>]*>", "a <body> section"),
        (r"</body\s*>", "a closing </body> tag"),
        (r"</html\s*>\s*$", "a closing </html> tag"),
    )
    for pattern, requirement in checks:
        if not re.search(pattern, html, flags=re.IGNORECASE | re.DOTALL):
            raise IncompleteHtmlError(
                f"Gemini returned incomplete HTML (missing {requirement})."
            )

    body_match = re.search(
        r"<body\b[^>]*>(.*?)</body\s*>",
        html,
        flags=re.IGNORECASE | re.DOTALL,
    )
    assert body_match is not None  # guaranteed by the document-boundary checks
    body_text = re.sub(r"<!--.*?-->|<[^>]*>", "", body_match.group(1), flags=re.DOTALL)
    if not unescape(body_text).replace("\xa0", "").strip():
        raise IncompleteHtmlError(
            "Gemini returned an HTML document with no visible body content."
        )

    # An unclosed style or script section swallows the rest of the document in
    # browsers and PDF renderers, which recreates the blank-output failure.
    for tag in ("style", "script"):
        openings = len(re.findall(rf"<{tag}\b[^>]*>", html, flags=re.IGNORECASE))
        closings = len(re.findall(rf"</{tag}\s*>", html, flags=re.IGNORECASE))
        if openings != closings:
            raise IncompleteHtmlError(
                f"Gemini returned incomplete HTML (unclosed <{tag}> section)."
            )

    return html


def _finish_reason(response: Any) -> str:
    """Return a readable model finish reason without relying on SDK internals."""
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        return "unknown"
    reason = getattr(candidates[0], "finish_reason", None)
    return getattr(reason, "value", str(reason or "unknown"))


def _generate_clean_html_openrouter(image_bytes: bytes, page: PageGeometry) -> str:
    """Use OpenRouter's vision-chat endpoint while retaining HTML validation."""
    request_image = image_bytes
    request_mime_type = "image/png"
    last_error: Exception | None = None
    for attempt in range(1, MAX_GENERATION_ATTEMPTS + 1):
        retry_instruction = ""
        if attempt > 1:
            retry_instruction = (
                " Return a more compact but complete HTML document. Finish all CSS "
                "rules and closing tags."
            )
        prompt = (
            "Analyse this prescription pad image and generate the editable HTML/CSS "
            "text-and-layout reconstruction. Reproduce every visible text line, rule, "
            "box, and colour region. Leave spaces for logos, seals, photos, signatures, "
            "and watermarks empty because their source graphics are overlaid separately. "
            "Return only a complete HTML file. "
            f"The required print page is {page.css_size} ({page.orientation}); set "
            "`.page` to exactly these dimensions and add no outer page margin or body padding."
            + retry_instruction
        )
        try:
            text = generate_openrouter_content(
                request_image,
                request_mime_type,
                SYSTEM_PROMPT,
                prompt,
                PRIMARY_OUTPUT_TOKENS if attempt == 1 else RETRY_OUTPUT_TOKENS,
            )
            return validate_reconstruction_html(_extract_html(text))
        except IncompleteHtmlError as exc:
            last_error = exc
        except Exception as exc:
            last_error = exc
            if attempt == MAX_GENERATION_ATTEMPTS:
                raise RuntimeError(f"OpenRouter reconstruction request failed. {exc}") from exc
        request_image = _compact_retry_image(image_bytes)
        request_mime_type = "image/jpeg"
    raise RuntimeError(
        "OpenRouter could not produce a complete HTML document after "
        f"{MAX_GENERATION_ATTEMPTS} attempts. {last_error}"
    )


def generate_clean_html(image_bytes: bytes, page: PageGeometry) -> str:
    """
    Send a prescription pad image to Gemini 3.6 Flash and return a
    self-contained HTML/CSS reproduction of the pad.
    """
    if use_openrouter():
        return _generate_clean_html_openrouter(image_bytes, page)
    client = genai.Client(api_key=get_gemini_api_key())

    # Gemini 3 uses thinking levels rather than 2.5's token budgets. Medium
    # preserves the spatial and visual reasoning needed for faithful document
    # reconstruction without escalating to the slowest setting.
    last_error: Exception | None = None
    response: Any = None
    request_image = image_bytes
    request_mime_type = "image/png"
    rate_limit_waited = False
    for attempt in range(1, MAX_GENERATION_ATTEMPTS + 1):
        output_tokens = (
            PRIMARY_OUTPUT_TOKENS if attempt == 1 else RETRY_OUTPUT_TOKENS
        )
        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            max_output_tokens=output_tokens,
            thinking_config=types.ThinkingConfig(thinking_level="medium"),
            http_options=types.HttpOptions(timeout=GENERATION_TIMEOUT_MS),
        )
        retry_instruction = ""
        if attempt > 1:
            retry_instruction = (
                " Your previous attempt did not complete. Return "
                "a more compact, complete HTML document now; finish every CSS "
                "rule and all closing tags."
            )

        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=[
                    types.Part.from_bytes(
                        data=request_image,
                        mime_type=request_mime_type,
                    ),
                    (
                        "Analyse this prescription pad image carefully and generate "
                        "the editable HTML/CSS text-and-layout reconstruction. Reproduce "
                        "every visible text line, rule, box, and colour region. Leave "
                        "the spaces for logos, seals, photos, signatures, and watermarks "
                        "empty: those exact source graphics are overlaid in a separate "
                        "stage. The output must look like "
                        "a professionally printed pad — clean, crisp, and identical "
                        "to the original layout. Return only the complete HTML file. "
                        f"The required print page is {page.css_size} "
                        f"({page.orientation}); set `.page` to exactly these dimensions "
                        "and do not add any outer page margin or body padding."
                        + retry_instruction
                    ),
                ],
                config=config,
            )
        except Exception as exc:
            last_error = exc
            if _is_rate_limit_error(exc):
                retry_delay = _retry_delay_seconds(exc)
                if (
                    not rate_limit_waited
                    and retry_delay is not None
                    and retry_delay <= MAX_RATE_LIMIT_WAIT_SECONDS
                    and attempt < MAX_GENERATION_ATTEMPTS
                ):
                    rate_limit_waited = True
                    time.sleep(retry_delay)
                    continue
                wait_hint = (
                    f" Gemini requested a {retry_delay}-second delay."
                    if retry_delay is not None
                    else ""
                )
                raise GeminiRateLimitError(
                    "Gemini API quota is currently exhausted for this key."
                    f"{wait_hint} Use a key with available quota or try again later."
                ) from exc
            if not _is_transient_api_error(exc) or attempt == MAX_GENERATION_ATTEMPTS:
                raise RuntimeError(
                    "Gemini reconstruction request failed. "
                    f"{exc}"
                ) from exc

            request_image = _compact_retry_image(image_bytes)
            request_mime_type = "image/jpeg"
            time.sleep(TRANSIENT_RETRY_DELAYS_SECONDS[attempt - 1])
            continue

        html = _extract_html(getattr(response, "text", "") or "")
        try:
            return validate_reconstruction_html(html)
        except IncompleteHtmlError as exc:
            last_error = exc
            request_image = _compact_retry_image(image_bytes)
            request_mime_type = "image/jpeg"

    reason = _finish_reason(response)
    raise RuntimeError(
        "Gemini could not produce a complete HTML document after "
        f"{MAX_GENERATION_ATTEMPTS} attempts (last finish reason: {reason}). "
        f"{last_error}"
    )
