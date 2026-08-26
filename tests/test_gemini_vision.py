"""Offline regression tests for Gemini response handling.

These tests never call Gemini. They ensure a response cut off mid-stylesheet
cannot reach the HTML renderer again.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from pipeline import gemini_vision
from pipeline.page_geometry import PageGeometry


COMPLETE_HTML = """<!DOCTYPE html>
<html><head><style>body { color: #123; }</style></head>
<body><main>Prescription</main></body></html>"""
TRUNCATED_HTML = """<!DOCTYPE html>
<html><head><style>body { color: #123; }"""
EMPTY_BODY_HTML = """<!DOCTYPE html>
<html><head><style>body { color: #123; }</style></head>
<body>&nbsp;</body></html>"""


class _FakeModels:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response


class GeminiVisionTests(unittest.TestCase):
    def test_validator_rejects_truncated_document(self):
        with self.assertRaisesRegex(
            gemini_vision.IncompleteHtmlError, "closing </head> tag"
        ):
            gemini_vision.validate_reconstruction_html(TRUNCATED_HTML)

    def test_extract_html_removes_fences_and_trailing_prose(self):
        raw = f"```html\n{COMPLETE_HTML}\n```\nThis is the result."
        self.assertEqual(gemini_vision._extract_html(raw), COMPLETE_HTML)

    def test_validator_rejects_visually_empty_document(self):
        with self.assertRaisesRegex(
            gemini_vision.IncompleteHtmlError, "no visible body content"
        ):
            gemini_vision.validate_reconstruction_html(EMPTY_BODY_HTML)

    @patch("pipeline.gemini_vision.use_openrouter", return_value=False)
    @patch("pipeline.gemini_vision.get_gemini_api_key", return_value="test-key")
    @patch("pipeline.gemini_vision.genai.Client")
    def test_generation_retries_incomplete_response(self, client_class, _key, _openrouter):
        first = SimpleNamespace(
            text=TRUNCATED_HTML,
            candidates=[SimpleNamespace(finish_reason="MAX_TOKENS")],
        )
        second = SimpleNamespace(
            text=COMPLETE_HTML,
            candidates=[SimpleNamespace(finish_reason="STOP")],
        )
        models = _FakeModels([first, second])
        client_class.return_value = SimpleNamespace(models=models)

        page = PageGeometry.from_pixels(1489, 1056)
        html = gemini_vision.generate_clean_html(b"test image", page=page)

        self.assertEqual(html, COMPLETE_HTML)
        self.assertEqual(len(models.calls), 2)
        config = models.calls[0]["config"]
        self.assertEqual(config.max_output_tokens, 16384)
        self.assertEqual(config.thinking_config.thinking_level.value, "MEDIUM")
        self.assertEqual(config.http_options.timeout, 150000)
        self.assertEqual(models.calls[1]["config"].max_output_tokens, 24576)
        self.assertIn("previous attempt did not complete", models.calls[1]["contents"][1])
        self.assertIn(page.css_size, models.calls[0]["contents"][1])

    @patch("pipeline.gemini_vision.use_openrouter", return_value=False)
    @patch("pipeline.gemini_vision.time.sleep")
    @patch("pipeline.gemini_vision.get_gemini_api_key", return_value="test-key")
    @patch("pipeline.gemini_vision.genai.Client")
    def test_generation_retries_transient_deadline_with_compact_image(
        self, client_class, _key, sleep, _openrouter
    ):
        models = _FakeModels([
            RuntimeError("504 DEADLINE_EXCEEDED"),
            SimpleNamespace(text=COMPLETE_HTML, candidates=[]),
        ])
        client_class.return_value = SimpleNamespace(models=models)

        html = gemini_vision.generate_clean_html(
            b"not-a-real-image", page=PageGeometry.from_pixels(100, 200)
        )

        self.assertEqual(html, COMPLETE_HTML)
        self.assertEqual(len(models.calls), 2)
        sleep.assert_called_once_with(4)
        self.assertEqual(
            models.calls[0]["contents"][0].inline_data.mime_type, "image/png"
        )
        self.assertEqual(
            models.calls[1]["contents"][0].inline_data.mime_type, "image/jpeg"
        )

    @patch("pipeline.gemini_vision.use_openrouter", return_value=False)
    @patch("pipeline.gemini_vision.time.sleep")
    @patch("pipeline.gemini_vision.get_gemini_api_key", return_value="test-key")
    @patch("pipeline.gemini_vision.genai.Client")
    def test_generation_honors_gemini_rate_limit_delay(
        self, client_class, _key, sleep, _openrouter
    ):
        models = _FakeModels([
            RuntimeError("429 RESOURCE_EXHAUSTED: retryDelay: '46s'"),
            SimpleNamespace(text=COMPLETE_HTML, candidates=[]),
        ])
        client_class.return_value = SimpleNamespace(models=models)

        html = gemini_vision.generate_clean_html(
            b"not-a-real-image", page=PageGeometry.from_pixels(100, 200)
        )

        self.assertEqual(html, COMPLETE_HTML)
        self.assertEqual(len(models.calls), 2)
        sleep.assert_any_call(46)
        self.assertEqual(
            models.calls[1]["contents"][0].inline_data.mime_type, "image/png"
        )

    @patch("pipeline.gemini_vision.use_openrouter", return_value=False)
    @patch("pipeline.gemini_vision.time.sleep")
    @patch("pipeline.gemini_vision.get_gemini_api_key", return_value="test-key")
    @patch("pipeline.gemini_vision.genai.Client")
    def test_generation_uses_free_tier_fallback_after_primary_503(
        self, client_class, _key, _sleep, _openrouter
    ):
        models = _FakeModels([
            RuntimeError("503 UNAVAILABLE: temporary high demand"),
            RuntimeError("503 UNAVAILABLE: temporary high demand"),
            RuntimeError("503 UNAVAILABLE: temporary high demand"),
            SimpleNamespace(text=COMPLETE_HTML, candidates=[]),
        ])
        client_class.return_value = SimpleNamespace(models=models)

        html = gemini_vision.generate_clean_html(
            b"test image", page=PageGeometry.from_pixels(100, 200)
        )

        self.assertEqual(html, COMPLETE_HTML)
        self.assertEqual(
            [call["model"] for call in models.calls],
            ["gemini-3.6-flash"] * 3 + ["gemini-3.5-flash"],
        )

if __name__ == "__main__":
    unittest.main()
