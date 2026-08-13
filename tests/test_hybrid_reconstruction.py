"""Regression coverage for the editable-text/source-artwork pipeline."""

import unittest
import base64
import io
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

from pipeline.asset_reconstruction import (
    complete_graphic_bounds,
    reconstruct_assets,
    sanitize_graphics,
)
from pipeline.document_analyzer import analyze_graphics
from pipeline.document_manifest import DocumentManifest
from pipeline.page_geometry import PageGeometry
from pipeline.template_renderer import inject_source_graphics


SOURCE_HTML = """<!DOCTYPE html><html><head><style>
.page { width: 100mm; height: 150mm; }
</style></head><body><div class="page"><h1>Editable clinic title</h1></div></body></html>"""


class _FakeModels:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


class HybridReconstructionTests(unittest.TestCase):
    def test_graphics_guard_recovers_complete_logo_and_watermark(self):
        image = Image.new("RGB", (512, 714), "white")
        pixels = np.asarray(image).copy()
        cv2.circle(pixels, (310, 476), 138, (175, 175, 175), 3)
        cv2.circle(pixels, (60, 55), 35, (20, 20, 20), 3)
        image = Image.fromarray(pixels)
        graphics = DocumentManifest.from_dict(
            {
                "elements": [
                    {
                        "id": "logo",
                        "kind": "image",
                        "role": "logo",
                        "bbox": [0.027, 0.05, 0.098, 0.123],
                    },
                    {
                        "id": "seal",
                        "kind": "image",
                        "role": "watermark_seal",
                        "bbox": [0.455, 0.336, 0.545, 0.388],
                    },
                ]
            }
        )

        corrected = complete_graphic_bounds(image, graphics)
        logo, watermark = corrected.elements

        self.assertGreater(logo.box.width, 0.12)
        self.assertLess(logo.box.y, 0.05)
        self.assertLess(watermark.box.x, 0.40)
        self.assertGreater(watermark.box.y, 0.42)
        self.assertLess(watermark.box.x + watermark.box.width, 0.93)

    def test_graphics_guard_uses_analysis_coordinates_for_high_resolution_assets(self):
        """Normalized coordinates can be detected at low resolution and cropped at print DPI."""
        analysis = Image.new("RGB", (512, 714), "white")
        pixels = np.asarray(analysis).copy()
        cv2.circle(pixels, (310, 476), 138, (175, 175, 175), 3)
        analysis = Image.fromarray(pixels)
        graphics = DocumentManifest.from_dict(
            {
                "elements": [
                    {
                        "id": "seal",
                        "kind": "image",
                        "role": "watermark",
                        "bbox": [0.455, 0.336, 0.545, 0.388],
                    }
                ]
            }
        )

        corrected = complete_graphic_bounds(analysis, graphics).elements[0]

        self.assertLess(corrected.box.x, 0.40)
        self.assertGreater(corrected.box.y, 0.42)

    def test_source_artwork_uses_transparent_logo_and_watermark_layers(self):
        image = Image.new("RGB", (512, 714), "white")
        pixels = np.asarray(image).copy()
        cv2.circle(pixels, (60, 57), 34, (20, 20, 20), 3)
        cv2.circle(pixels, (310, 476), 138, (175, 175, 175), 3)
        image = Image.fromarray(pixels)
        graphics = DocumentManifest.from_dict(
            {
                "elements": [
                    {"id": "logo", "kind": "image", "role": "logo", "bbox": [0.03, 0.05, 0.1, 0.12]},
                    {"id": "seal", "kind": "image", "role": "watermark", "bbox": [0.32, 0.44, 0.58, 0.43]},
                ]
            }
        )
        graphics = complete_graphic_bounds(image, graphics)
        assets = reconstruct_assets(image, graphics)

        logo = Image.open(io.BytesIO(base64.b64decode(assets["logo"].split(",", 1)[1]))).convert("RGBA")
        watermark = Image.open(io.BytesIO(base64.b64decode(assets["seal"].split(",", 1)[1]))).convert("RGBA")

        self.assertEqual(logo.getchannel("A").getpixel((0, 0)), 0)
        self.assertGreater(watermark.getchannel("A").getextrema()[1], 0)
        self.assertLessEqual(watermark.getchannel("A").getextrema()[1], 97)

    def test_photo_watermark_keeps_its_proposed_bounds_and_soft_alpha(self):
        image = Image.new("RGB", (512, 714), "white")
        pixels = np.asarray(image).copy()
        pixels[300:560, 160:410] = (220, 220, 220)
        pixels[350:520, 215:365] = (168, 168, 168)
        image = Image.fromarray(pixels)
        graphics = DocumentManifest.from_dict(
            {
                "elements": [
                    {
                        "id": "baby-photo",
                        "kind": "image",
                        "role": "watermark_photo",
                        "bbox": [0.31, 0.42, 0.49, 0.36],
                    }
                ]
            }
        )

        corrected = complete_graphic_bounds(image, graphics).elements[0]
        asset = reconstruct_assets(image, DocumentManifest(elements=(corrected,)))["baby-photo"]
        photo = Image.open(io.BytesIO(base64.b64decode(asset.split(",", 1)[1]))).convert("RGBA")

        self.assertAlmostEqual(corrected.box.x, 0.298, places=3)
        self.assertAlmostEqual(corrected.box.y, 0.406, places=3)
        self.assertLessEqual(photo.getchannel("A").getextrema()[1], 72)

    def test_graphic_quality_gate_rejects_header_text_labeled_as_watermark(self):
        graphics = DocumentManifest.from_dict(
            {
                "elements": [
                    {
                        "id": "bad-header",
                        "kind": "image",
                        "role": "watermark_photo",
                        "bbox": [0.2, 0.05, 0.6, 0.12],
                    },
                    {
                        "id": "portrait",
                        "kind": "image",
                        "role": "watermark_photo",
                        "bbox": [0.2, 0.28, 0.55, 0.4],
                    },
                    {
                        "id": "caduceus",
                        "kind": "image",
                        "role": "medical_icon",
                        "bbox": [0.45, 0.04, 0.1, 0.16],
                    },
                ]
            }
        )

        accepted, warnings = sanitize_graphics(graphics)

        self.assertEqual([element.id for element in accepted.elements], ["portrait", "caduceus"])
        self.assertEqual(len(warnings), 1)

    def test_source_artwork_layers_are_inserted_inside_the_page(self):
        graphics = DocumentManifest.from_dict(
            {
                "background": "#ffffff",
                "elements": [
                    {
                        "id": "seal",
                        "kind": "image",
                        "role": "watermark",
                        "bbox": [0.2, 0.3, 0.4, 0.4],
                        "opacity": 0.18,
                    },
                    {
                        "id": "logo",
                        "kind": "image",
                        "role": "logo",
                        "bbox": [0.05, 0.05, 0.1, 0.1],
                    },
                ],
            }
        )

        html = inject_source_graphics(
            SOURCE_HTML,
            graphics,
            {"seal": "data:image/png;base64,c2VhbA==", "logo": "data:image/png;base64,bG9nbw=="},
            PageGeometry(100, 150),
        )

        page_start = html.index('<div class="page">')
        content_start = html.index("<h1>Editable clinic title</h1>")
        watermark_layer = html.index('<div class="source-graphics-layer source-watermark-layer"')
        artwork_layer = html.index('<div class="source-graphics-layer source-artwork-layer"')
        self.assertLess(page_start, watermark_layer)
        self.assertLess(watermark_layer, content_start)
        self.assertLess(artwork_layer, content_start)
        self.assertIn("source-graphics-contract", html)
        self.assertIn("source-watermark", html)
        self.assertNotIn('class="source-graphic-overlay source-artwork" src="data:image/png;base64,c2VhbA=="', html)
        self.assertIn("source-artwork", html)
        self.assertIn(".source-watermark-layer { z-index:10", html)
        self.assertIn(":not(.source-graphics-layer) { position:relative; z-index:20", html)
        self.assertIn("object-fit:contain", html)
        self.assertIn("svg.header-logo, svg.watermark-bg", html)

    @patch("pipeline.document_analyzer.get_gemini_api_key", return_value="test-key")
    @patch("pipeline.document_analyzer.genai.Client")
    def test_graphics_analysis_filters_response_to_image_elements(self, client_class, _key):
        response = SimpleNamespace(
            text="""{
              "background": "#ffffff",
              "elements": [
                {"id": "logo", "kind": "image", "role": "logo", "bbox": [0.1, 0.1, 0.2, 0.2]},
                {"id": "title", "kind": "text", "text": "Clinic", "bbox": [0.4, 0.1, 0.3, 0.1]}
              ]
            }"""
        )
        models = _FakeModels(response)
        client_class.return_value = SimpleNamespace(models=models)

        manifest = analyze_graphics(b"image", PageGeometry(100, 150))

        self.assertEqual([element.id for element in manifest.elements], ["logo"])
        self.assertEqual(models.calls[0]["config"].max_output_tokens, 6144)
        self.assertEqual(
            models.calls[0]["config"].thinking_config.thinking_level.value, "HIGH"
        )
        self.assertIn("Do not include any printed letters", models.calls[0]["contents"][1])


if __name__ == "__main__":
    unittest.main()
