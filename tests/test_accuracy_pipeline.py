"""Regression tests for the high-accuracy reconstruction stages."""

import io
import unittest

from PIL import Image, ImageDraw

from pipeline.asset_reconstruction import (
    complete_graphic_bounds,
    reconstruct_assets,
    sanitize_graphics,
)
from pipeline.document_manifest import DocumentManifest
from pipeline.page_geometry import PageGeometry
from pipeline.preprocess import preprocess_document
from pipeline.template_renderer import build_fidelity_html, render_manifest_html
from pipeline.visual_validation import compare_images


class AccuracyPipelineTests(unittest.TestCase):
    def test_preprocessing_rectifies_a_page_inside_a_dark_photo(self):
        photo = Image.new("RGB", (900, 700), "#20242b")
        draw = ImageDraw.Draw(photo)
        page = [(220, 25), (650, 70), (710, 660), (125, 610)]
        draw.polygon(page, fill="#eeeeea")
        for y in range(150, 560, 55):
            draw.line((210, y, 635, y + 25), fill="#343434", width=3)
        payload = io.BytesIO()
        photo.save(payload, format="PNG")

        result = preprocess_document(payload.getvalue())

        self.assertIsNotNone(result.page_corners)
        self.assertLess(result.canonical.width, photo.width)
        self.assertLess(result.canonical.height, photo.height)
        self.assertGreater(result.canonical.height, result.canonical.width)
        self.assertEqual(max(result.restored.size), 3508)

    def test_manifest_assets_are_source_crops_not_generated_shapes(self):
        manifest = DocumentManifest.from_dict(
            {
                "background": "#ffffff",
                "elements": [
                    {
                        "id": "logo",
                        "kind": "image",
                        "role": "logo",
                        "bbox": [0.1, 0.1, 0.25, 0.2],
                    },
                    {
                        "id": "title",
                        "kind": "text",
                        "bbox": [0.4, 0.1, 0.5, 0.05],
                        "text": "Clinic",
                    },
                ],
            }
        )
        source = Image.new("RGB", (600, 800), "white")
        draw = ImageDraw.Draw(source)
        draw.ellipse((60, 80, 210, 240), fill="#8b1e3f")

        assets = reconstruct_assets(source, manifest)
        html = render_manifest_html(
            manifest, assets, PageGeometry.from_pixels(*source.size)
        )

        self.assertIn("data:image/png;base64,", assets["logo"])
        self.assertIn('id="logo"', html)
        self.assertIn('id="title"', html)
        self.assertNotIn("<svg", html)

    def test_source_first_manifest_keeps_text_geometry_while_correcting_images(self):
        manifest = DocumentManifest.from_dict(
            {
                "background": "#ffffff",
                "elements": [
                    {
                        "id": "clinic-title",
                        "kind": "text",
                        "role": "title",
                        "bbox": [0.22, 0.05, 0.58, 0.06],
                        "text": "Exact Clinic Title",
                        "font_size": 0.028,
                        "font_weight": 700,
                    },
                    {
                        "id": "logo",
                        "kind": "image",
                        "role": "logo",
                        "bbox": [0.04, 0.04, 0.12, 0.12],
                    },
                ],
            }
        )
        source = Image.new("RGB", (600, 800), "white")
        ImageDraw.Draw(source).ellipse((24, 24, 108, 108), fill="#245d9f")

        checked, warnings = sanitize_graphics(manifest)
        corrected = complete_graphic_bounds(source, checked)
        title = next(element for element in corrected.elements if element.id == "clinic-title")
        assets = reconstruct_assets(source, corrected)
        html = render_manifest_html(
            corrected, assets, PageGeometry.from_pixels(*source.size)
        )

        self.assertEqual(warnings, ())
        self.assertEqual(title.box, manifest.elements[0].box)
        self.assertIn("Exact Clinic Title", html)
        self.assertIn('data-role="logo"', html)
        self.assertIn("object-fit:contain", html)

    def test_visual_score_distinguishes_identical_and_wrong_pages(self):
        source = Image.new("RGB", (500, 700), "white")
        draw = ImageDraw.Draw(source)
        draw.rectangle((40, 50, 460, 90), fill="black")
        identical = source.copy()
        wrong = Image.new("RGB", source.size, "white")

        exact_score = compare_images(source, identical)
        wrong_score = compare_images(source, wrong)

        self.assertGreater(exact_score.overall, 0.99)
        self.assertGreater(exact_score.overall, wrong_score.overall + 0.1)

    def test_fidelity_html_embeds_the_restored_page(self):
        source = Image.new("RGB", (200, 300), "#f8f8f4")
        html = build_fidelity_html(source, PageGeometry.from_pixels(200, 300))

        self.assertIn("data:image/png;base64,", html)
        self.assertIn("object-fit: fill", html)
        self.assertIn('class="page-image"', html)


if __name__ == "__main__":
    unittest.main()
