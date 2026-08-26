"""Provider-selection regression tests."""

from unittest.mock import patch

import config


def test_direct_gemini_key_takes_priority_over_openrouter() -> None:
    with (
        patch.dict("os.environ", {"AI_PROVIDER": ""}),
        patch.object(config.st, "secrets", {}),
        patch("config.get_gemini_api_key", return_value="gemini-key"),
        patch("config.get_openrouter_api_key", return_value="openrouter-key"),
    ):
        assert config.use_openrouter() is False


def test_openrouter_is_used_when_no_direct_gemini_key_exists() -> None:
    with (
        patch.dict("os.environ", {"AI_PROVIDER": ""}),
        patch.object(config.st, "secrets", {}),
        patch("config.get_gemini_api_key", side_effect=RuntimeError("missing key")),
        patch("config.get_openrouter_api_key", return_value="openrouter-key"),
    ):
        assert config.use_openrouter() is True
