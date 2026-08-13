"""Regression tests for the single-page PDF print contract."""

import unittest

from pipeline.html_render import prepare_print_html
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


if __name__ == "__main__":
    unittest.main()
