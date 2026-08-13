"""Tests for preserving an uploaded prescription's displayed page geometry."""

import unittest

from pipeline.page_geometry import PageGeometry


class PageGeometryTests(unittest.TestCase):
    def test_portrait_source_keeps_its_aspect_ratio(self):
        page = PageGeometry.from_pixels(1056, 1489)

        self.assertEqual(page.orientation, "portrait")
        self.assertAlmostEqual(page.height_mm, 297.0)
        self.assertAlmostEqual(page.width_mm / page.height_mm, 1056 / 1489)

    def test_landscape_source_keeps_its_orientation(self):
        page = PageGeometry.from_pixels(1489, 1056)

        self.assertEqual(page.orientation, "landscape")
        self.assertAlmostEqual(page.width_mm, 297.0)
        self.assertAlmostEqual(page.width_mm / page.height_mm, 1489 / 1056)


if __name__ == "__main__":
    unittest.main()
