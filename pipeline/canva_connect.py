"""Create a Canva editor link for a reconstructed prescription image.

The Canva Connect API creates a custom design containing the reconstructed PNG
as an editable image layer. In Canva the user can move, resize, crop, apply
effects to, and add further elements around that layer. Canva's Create Design
endpoint does not convert a rendered image into separately editable text or
vector layers; the app keeps the generated HTML download for source editing.
"""

import base64
import json
import math
import time
from io import BytesIO
from typing import Any

import requests
from PIL import Image

from config import CANVA_API_BASE, get_canva_token

MAX_CANVA_DIMENSION = 8_000
MAX_CANVA_AREA = 25_000_000
ASSET_UPLOAD_TIMEOUT_SECONDS = 90
POLL_INTERVAL_SECONDS = 2


class CanvaConnectError(RuntimeError):
    """A user-safe error returned by the Canva Connect API workflow."""


def _headers() -> dict[str, str]:
    token = get_canva_token()
    if not token:
        raise CanvaConnectError(
            "Canva is not configured. Add a CANVA_ACCESS_TOKEN and reconnect."
        )
    return {"Authorization": f"Bearer {token}"}


def _json_headers() -> dict[str, str]:
    headers = _headers()
    headers["Content-Type"] = "application/json"
    return headers


def _error_message(response: requests.Response, action: str) -> str:
    """Produce actionable API errors without exposing response internals."""
    message = ""
    try:
        payload = response.json()
        error = payload.get("error", payload) if isinstance(payload, dict) else {}
        if isinstance(error, dict):
            message = str(error.get("message") or error.get("code") or "")
    except ValueError:
        pass

    prefix = f"Canva could not {action} (HTTP {response.status_code})"
    return f"{prefix}: {message}" if message else prefix


def _request(
    method: str, url: str, *, action: str, **kwargs: Any
) -> requests.Response:
    """Call Canva with a bounded timeout and consistent user-facing errors."""
    try:
        response = requests.request(method, url, **kwargs)
    except requests.RequestException as exc:
        raise CanvaConnectError(f"Canva could not {action}: {exc}") from exc

    if not 200 <= response.status_code < 300:
        raise CanvaConnectError(_error_message(response, action))
    return response


def _asset_name(name: str) -> str:
    """Fit a name into Canva's 50-character asset-name limit."""
    return name.strip()[:50] or "prescription-pad"


def _design_dimensions(image_bytes: bytes) -> tuple[int, int]:
    """Return Canva-supported dimensions while preserving aspect ratio."""
    try:
        with Image.open(BytesIO(image_bytes)) as image:
            width, height = image.size
    except Exception as exc:
        raise CanvaConnectError("The reconstructed PNG could not be read.") from exc

    if width <= 0 or height <= 0:
        raise CanvaConnectError("The reconstructed PNG has invalid dimensions.")

    scale = min(
        1.0,
        MAX_CANVA_DIMENSION / max(width, height),
        math.sqrt(MAX_CANVA_AREA / (width * height)),
    )
    return max(40, round(width * scale)), max(40, round(height * scale))


def upload_asset(image_bytes: bytes, name: str = "prescription-pad") -> str:
    """Upload the reconstructed PNG and return its Canva asset ID."""
    url = f"{CANVA_API_BASE}/asset-uploads"
    metadata = {"name_base64": base64.b64encode(_asset_name(name).encode()).decode()}
    headers = _headers()
    headers["Content-Type"] = "application/octet-stream"
    headers["Asset-Upload-Metadata"] = json.dumps(metadata)

    response = _request(
        "POST",
        url,
        action="start the image upload",
        headers=headers,
        data=image_bytes,
        timeout=60,
    )
    job_id = response.json().get("job", {}).get("id")
    if not job_id:
        raise CanvaConnectError("Canva did not return an upload job ID.")

    deadline = time.monotonic() + ASSET_UPLOAD_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        time.sleep(POLL_INTERVAL_SECONDS)
        status_response = _request(
            "GET",
            f"{url}/{job_id}",
            action="check the image upload",
            headers=_headers(),
            timeout=30,
        )
        job = status_response.json().get("job", {})
        status = job.get("status")
        if status == "success":
            asset_id = job.get("asset", {}).get("id")
            if asset_id:
                return asset_id
            raise CanvaConnectError("Canva completed the upload without an asset ID.")
        if status == "failed":
            error = job.get("error", {})
            message = error.get("message") or error.get("code") or "unknown error"
            raise CanvaConnectError(f"Canva could not import the image: {message}")

    raise CanvaConnectError("Canva image upload timed out. Please try again.")


def create_design(
    asset_id: str,
    *,
    width: int,
    height: int,
    title: str = "Prescription Pad - Clean",
) -> dict[str, str]:
    """Create a correctly sized Canva design and return its temporary URLs."""
    url = f"{CANVA_API_BASE}/designs"
    payload = {
        "type": "type_and_asset",
        "design_type": {"type": "custom", "width": width, "height": height},
        "asset_id": asset_id,
        "title": title[:255],
    }
    response = _request(
        "POST",
        url,
        action="create the editable design",
        headers=_json_headers(),
        json=payload,
        timeout=60,
    )
    design = response.json().get("design", {})
    design_id = design.get("id")
    urls = design.get("urls", {})
    edit_url = urls.get("edit_url")
    if not design_id or not edit_url:
        raise CanvaConnectError("Canva created no editable design URL.")

    result = {"design_id": design_id, "edit_url": edit_url}
    view_url = urls.get("view_url")
    if view_url:
        result["view_url"] = view_url
    return result


def push_to_canva(
    image_bytes: bytes,
    title: str = "Prescription Pad - Clean",
) -> dict[str, str]:
    """Upload the PNG and create a same-aspect-ratio editable Canva design."""
    width, height = _design_dimensions(image_bytes)
    asset_id = upload_asset(image_bytes, name=title)
    design = create_design(asset_id, width=width, height=height, title=title)
    return {"asset_id": asset_id, "width": str(width), "height": str(height), **design}
