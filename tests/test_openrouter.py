"""Offline contract tests for the OpenRouter vision adapter."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from pipeline.openrouter import generate_openrouter_content


class OpenRouterTests(unittest.TestCase):
    @patch("pipeline.openrouter.get_openrouter_api_key", return_value="test-key")
    @patch("pipeline.openrouter.requests.post")
    def test_sends_a_base64_vision_request_and_returns_completion(self, post, _key):
        response = SimpleNamespace(
            ok=True,
            json=lambda: {"choices": [{"message": {"content": "<html></html>"}}]},
        )
        post.return_value = response

        result = generate_openrouter_content(
            b"image-data",
            "image/png",
            "system prompt",
            "user prompt",
            1000,
        )

        self.assertEqual(result, "<html></html>")
        _, kwargs = post.call_args
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer test-key")
        self.assertEqual(kwargs["json"]["model"], "google/gemini-2.5-flash")
        image = kwargs["json"]["messages"][1]["content"][1]["image_url"]["url"]
        self.assertEqual(image, "data:image/png;base64,aW1hZ2UtZGF0YQ==")


if __name__ == "__main__":
    unittest.main()
