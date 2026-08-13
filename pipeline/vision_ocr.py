import base64
from typing import Any, Dict, List

import requests

from config import VISION_ENDPOINT, get_vision_api_key


def _call_vision(image_bytes: bytes) -> Dict[str, Any]:
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    payload = {
        "requests": [
            {
                "image": {"content": b64},
                "features": [
                    {"type": "DOCUMENT_TEXT_DETECTION"},
                    {"type": "IMAGE_PROPERTIES"},
                ],
            }
        ]
    }
    resp = requests.post(
        VISION_ENDPOINT,
        params={"key": get_vision_api_key()},
        json=payload,
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    response = data["responses"][0]
    if "error" in response:
        raise RuntimeError(f"Vision API error: {response['error']}")
    return response


def _block_text(block: Dict[str, Any]) -> str:
    words = []
    for para in block.get("paragraphs", []):
        for word in para.get("words", []):
            chars = [s.get("text", "") for s in word.get("symbols", [])]
            words.append("".join(chars))
    return " ".join(words)


def _bbox(block: Dict[str, Any]) -> Dict[str, int]:
    verts = block["boundingBox"]["vertices"]
    xs = [v.get("x", 0) for v in verts]
    ys = [v.get("y", 0) for v in verts]
    return {"x0": min(xs), "y0": min(ys), "x1": max(xs), "y1": max(ys)}


def extract_manifest(image_bytes: bytes) -> Dict[str, Any]:
    """
    Calls Google Vision DOCUMENT_TEXT_DETECTION and reduces the result to a
    simple manifest used for reconstruction:
      { full_text, lines, header_lines, footer_lines, page_height }
    header_lines / footer_lines are picked by vertical position on the page
    (top ~25% / bottom ~15%) since prescription pads are consistently laid
    out with clinic info on top and address/reg-no on the bottom.
    """
    result = _call_vision(image_bytes)
    annotation = result.get("fullTextAnnotation", {})
    full_text = annotation.get("text", "")

    lines: List[Dict[str, Any]] = []
    pages = annotation.get("pages", [])
    page_height = pages[0].get("height", 1000) if pages else 1000

    if pages:
        for block in pages[0].get("blocks", []):
            text = _block_text(block).strip()
            if not text:
                continue
            bbox = _bbox(block)
            lines.append({"text": text, "bbox": bbox, "height": bbox["y1"] - bbox["y0"]})
        lines.sort(key=lambda l: l["bbox"]["y0"])

    header_cut = page_height * 0.25
    footer_cut = page_height * 0.85

    header_lines = [l for l in lines if l["bbox"]["y0"] < header_cut]
    footer_lines = [l for l in lines if l["bbox"]["y0"] > footer_cut]

    return {
        "full_text": full_text,
        "lines": lines,
        "header_lines": header_lines,
        "footer_lines": footer_lines,
        "page_height": page_height,
    }
