"""Fast-path tests for image preprocessing."""

import io
import unittest

from PIL import Image

from pipeline.preprocess import MAX_ANALYSIS_DIMENSION, preprocess_document, preprocess_image


class PreprocessTests(unittest.TestCase):
    def test_oversized_photo_is_capped_for_fast_vision_analysis(self):
        source = Image.new("RGB", (3072, 2304), "white")
        buffer = io.BytesIO()
        source.save(buffer, format="JPEG", quality=80)

        result = preprocess_image(buffer.getvalue())

        self.assertEqual(result.size, (2048, 1536))
        self.assertEqual(max(result.size), MAX_ANALYSIS_DIMENSION)

    def test_manual_crop_excludes_camera_background_and_reports_full_confidence(self):
        source = Image.new("RGB", (1000, 1400), "#20242b")
        page = Image.new("RGB", (800, 1100), "#f8f8f2")
        source.paste(page, (100, 180))
        buffer = io.BytesIO()
        source.save(buffer, format="PNG")

        result = preprocess_document(buffer.getvalue(), manual_crop=(0.1, 0.13, 0.9, 0.92))

        self.assertEqual(result.page_detection_method, "manual-crop")
        self.assertEqual(result.page_confidence, 1.0)
        self.assertLess(result.canonical.width, source.width)
        self.assertLess(result.canonical.height, source.height)

if __name__ == "__main__":
    unittest.main()
