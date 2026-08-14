"""Tests for manual Gemini-key selection."""

from unittest.mock import patch

import config


def test_selected_saved_key_is_used_when_no_session_override() -> None:
    with (
        patch("config.st.session_state", {"gemini_saved_key_index": 2}),
        patch("config.get_configured_gemini_api_keys", return_value=("key-one", "key-two")),
    ):
        assert config.get_gemini_api_key() == "key-two"


def test_direct_session_key_takes_priority_over_selected_saved_key() -> None:
    with (
        patch(
            "config.st.session_state",
            {"gemini_api_key": "session-key", "gemini_saved_key_index": 2},
        ),
        patch("config.get_configured_gemini_api_keys", return_value=("key-one", "key-two")),
    ):
        assert config.get_gemini_api_key() == "session-key"


def test_first_saved_key_is_the_default_when_a_pool_is_configured() -> None:
    with (
        patch("config.st.session_state", {}),
        patch("config.get_configured_gemini_api_keys", return_value=("key-one", "key-two")),
    ):
        assert config.get_gemini_api_key() == "key-one"
