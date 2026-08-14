"""Regression tests for the single-page PDF print contract."""

import unittest
from unittest.mock import patch

from pipeline.html_render import prepare_print_html, render_html
from pipeline.page_geometry import PageGeometry


SCREEN_STYLE_HTML = """<!DOCTYPE html><html><head><style>
body { display: flex; padding: 20px; }
.page { width: 210mm; height: 297mm; margin: 10mm auto; }
</style></head><body><div class="page">Prescription</div></body></html>"""


class HtmlRenderTests(unittest.TestCase):
    def test_print_contract_overrides_screen_margins_and_page_size(self):
        page = PageGeometry.from_pixels(1489, 1056)

        printable = prepare_print_html(SCREEN_STYLE_HTML, page)

        self.assertIn('@page {', printable)
        self.assertIn(f'size: {page.css_size};', printable)
        self.assertIn('padding: 0 !important;', printable)
        self.assertIn(f'width: {page.width_mm:.3f}mm !important;', printable)
        self.assertIn(f'height: {page.height_mm:.3f}mm !important;', printable)
        self.assertIn('body > .page .patient-info .field', printable)
        self.assertEqual(printable.count('prescription-print-contract'), 1)

    @patch("pipeline.html_render._render_with_weasyprint")
    @patch("pipeline.html_render._render_with_playwright")
    def test_chromium_is_preferred_for_consistent_multilingual_rendering(
        self, playwright, weasyprint
    ):
        page = PageGeometry.from_pixels(100, 200)
        playwright.return_value = ("chromium-image", b"chromium-pdf")

        result = render_html(SCREEN_STYLE_HTML, page=page)

        self.assertEqual(result, ("chromium-image", b"chromium-pdf"))
        playwright.assert_called_once()
        weasyprint.assert_not_called()


if __name__ == "__main__":
    unittest.main()
