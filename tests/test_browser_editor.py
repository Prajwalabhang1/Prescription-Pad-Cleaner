"""Offline tests for the no-account browser editor output."""

import tempfile
import unittest
from pathlib import Path

from pipeline.browser_editor import (
    BrowserEditorError,
    EDITOR_MARKER,
    build_editor_html,
    publish_editor_html,
)


SOURCE_HTML = """<!DOCTYPE html><html><head><style>.page { color: #123; }</style></head>
<body><div class="page"><h1>Clinic</h1><img src="logo.png"></div></body></html>"""


class BrowserEditorTests(unittest.TestCase):
    def test_builds_self_contained_editor_controls(self):
        editor_html = build_editor_html(SOURCE_HTML)

        self.assertIn(EDITOR_MARKER, editor_html)
        self.assertIn("Download edited HTML", editor_html)
        self.assertIn("Click text to edit", editor_html)
        self.assertIn("<h1>Clinic</h1>", editor_html)

    def test_publish_uses_hashed_non_sensitive_filename(self):
        editor_html = build_editor_html(SOURCE_HTML)
        with tempfile.TemporaryDirectory() as directory:
            url = publish_editor_html(editor_html, Path(directory))
            output = next(Path(directory).iterdir())

            self.assertRegex(url, r"^/app/static/editor/prescription-[0-9a-f]{20}\.html$")
            self.assertEqual(output.read_text(encoding="utf-8"), editor_html)

    def test_rejects_incomplete_documents(self):
        with self.assertRaises(BrowserEditorError):
            build_editor_html("<html><head></head><body>unfinished")


if __name__ == "__main__":
    unittest.main()
