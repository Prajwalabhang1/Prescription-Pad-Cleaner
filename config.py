import os

import streamlit as st


# ---------------------------------------------------------------------------
# Gemini
# ---------------------------------------------------------------------------
def get_gemini_api_key() -> str:
    """Retrieve a session override first, then env or Streamlit secrets."""
    session_key = st.session_state.get("gemini_api_key", "").strip()
    if session_key:
        return session_key
    key = os.environ.get("GOOGLE_API_KEY")
    if key:
        return key
    try:
        return st.secrets["GOOGLE_API_KEY"]
    except Exception:
        raise RuntimeError(
            "GOOGLE_API_KEY not found. Set it as an env var or in "
            ".streamlit/secrets.toml"
        )


# Gemini 2.5 Flash is unavailable to new Gemini API users. Gemini 3.6 Flash
# is Google's current production replacement with stronger multimodal and
# spatial reasoning for document reconstruction.
GEMINI_MODEL = "gemini-3.6-flash"

# OpenRouter provides a single OpenAI-compatible gateway to vision-capable
# models. It takes priority when a key is supplied, allowing the app to switch
# providers without changing reconstruction code or storing a key on disk.
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "google/gemini-2.5-flash")


def get_openrouter_api_key() -> str:
    """Retrieve an OpenRouter key from the active browser session or env."""
    session_key = st.session_state.get("openrouter_api_key", "").strip()
    if session_key:
        return session_key
    return os.environ.get("OPENROUTER_API_KEY", "").strip()


def use_openrouter() -> bool:
    """Whether AI requests should use OpenRouter for this session."""
    return bool(get_openrouter_api_key())

# ---------------------------------------------------------------------------
# Canva Connect API
# ---------------------------------------------------------------------------
CANVA_API_BASE = "https://api.canva.com/rest/v1"


def get_canva_token() -> str:
    """Retrieve the Canva Connect OAuth token from env or Streamlit secrets."""
    key = os.environ.get("CANVA_ACCESS_TOKEN")
    if key:
        return key
    try:
        return st.secrets["CANVA_ACCESS_TOKEN"]
    except Exception:
        return ""  # gracefully degrade — Canva features just won't be available


# ---------------------------------------------------------------------------
# Canvas / page settings
# ---------------------------------------------------------------------------
CANVAS_SIZE = (1240, 1754)  # A4 @ ~150 dpi
