"""Offline contract tests for the Canva editable-design workflow."""

import base64
import json
import unittest
from io import BytesIO
from unittest.mock import patch

from PIL import Image

from pipeline import canva_connect


class _Response:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


def _png_bytes(size=(1244, 1754)):
    image = Image.new("RGB", size, "white")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


class CanvaConnectTests(unittest.TestCase):
    def test_design_dimensions_preserve_aspect_ratio_within_canva_limits(self):
        width, height = canva_connect._design_dimensions(_png_bytes((1244, 1754)))
        self.assertEqual((width, height), (1244, 1754))

        width, height = canva_connect._design_dimensions(_png_bytes((10_000, 100)))
        self.assertEqual((width, height), (8000, 80))

    @patch("pipeline.canva_connect.get_canva_token", return_value="token")
    @patch("pipeline.canva_connect.requests.request")
    def test_create_design_uses_current_canva_schema_and_output_size(
        self, request, _token
    ):
        request.return_value = _Response(
            200,
            {
                "design": {
                    "id": "design-1",
                    "urls": {"edit_url": "https://canva.test/edit", "view_url": "https://canva.test/view"},
                }
            },
        )

        result = canva_connect.create_design(
            "asset-1", width=1244, height=1754, title="Clean Pad"
        )

        self.assertEqual(result["edit_url"], "https://canva.test/edit")
        payload = request.call_args.kwargs["json"]
        self.assertEqual(payload["type"], "type_and_asset")
        self.assertEqual(payload["asset_id"], "asset-1")
        self.assertEqual(payload["design_type"], {"type": "custom", "width": 1244, "height": 1754})

    @patch("pipeline.canva_connect.get_canva_token", return_value="token")
    @patch("pipeline.canva_connect.time.sleep")
    @patch("pipeline.canva_connect.requests.request")
    def test_upload_asset_polls_to_success_and_sends_metadata(
        self, request, _sleep, _token
    ):
        request.side_effect = [
            _Response(200, {"job": {"id": "job-1", "status": "in_progress"}}),
            _Response(200, {"job": {"id": "job-1", "status": "success", "asset": {"id": "asset-1"}}}),
        ]

        asset_id = canva_connect.upload_asset(b"png", name="Clean Pad")

        self.assertEqual(asset_id, "asset-1")
        metadata = json.loads(request.call_args_list[0].kwargs["headers"]["Asset-Upload-Metadata"])
        self.assertEqual(base64.b64decode(metadata["name_base64"]).decode(), "Clean Pad")

    @patch("pipeline.canva_connect.create_design")
    @patch("pipeline.canva_connect.upload_asset", return_value="asset-1")
    def test_push_to_canva_uses_reconstructed_png_dimensions(self, upload, create):
        create.return_value = {"design_id": "design-1", "edit_url": "https://canva.test/edit"}

        result = canva_connect.push_to_canva(_png_bytes(), title="Clean Pad")

        upload.assert_called_once()
        create.assert_called_once_with(
            "asset-1", width=1244, height=1754, title="Clean Pad"
        )
        self.assertEqual(result["edit_url"], "https://canva.test/edit")


if __name__ == "__main__":
    unittest.main()
