import os

import streamlit as st


# ---------------------------------------------------------------------------
# Gemini
# ---------------------------------------------------------------------------
def get_configured_gemini_api_keys() -> tuple[str, ...]:
    """Return manually selectable Gemini keys from local Streamlit secrets."""
    try:
        raw_keys = st.secrets.get("GEMINI_API_KEYS", ())
    except Exception:
        raw_keys = ()

    if isinstance(raw_keys, str):
        raw_keys = (raw_keys,)

    keys: list[str] = []
    for key in raw_keys:
        normalized = str(key).strip()
        if normalized and normalized not in keys:
            keys.append(normalized)
    return tuple(keys)


def get_gemini_api_key() -> str:
    """Retrieve a session override first, then env or Streamlit secrets."""
    session_key = st.session_state.get("gemini_api_key", "").strip()
    if session_key:
        return session_key

    configured_keys = get_configured_gemini_api_keys()
    selected_index = st.session_state.get("gemini_saved_key_index", 0)
    if isinstance(selected_index, int) and 1 <= selected_index <= len(configured_keys):
        return configured_keys[selected_index - 1]

    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if key:
        return key
    try:
        return st.secrets.get("GEMINI_API_KEY") or st.secrets["GOOGLE_API_KEY"]
    except Exception:
        raise RuntimeError(
            "GOOGLE_API_KEY not found. Set it as an env var or in "
            ".streamlit/secrets.toml"
        )


# Gemini 2.5 Flash is unavailable to new Gemini API users. Gemini 3.6 Flash
# is Google's current production replacement with stronger multimodal and
# spatial reasoning for document reconstruction.
GEMINI_MODEL = "gemini-3.6-flash"
# Use the next strongest free-tier multimodal option only when the primary
# endpoint is temporarily unavailable. The primary remains the accuracy-first
# choice for every normal reconstruction.
GEMINI_FALLBACK_MODELS = ("gemini-3.5-flash",)
GEMINI_MODELS = (GEMINI_MODEL, *GEMINI_FALLBACK_MODELS)

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
    """Use OpenRouter only when no direct Gemini key has been configured.

    A user-provided Gemini key is an explicit provider choice. This avoids a
    lingering OpenRouter environment variable silently taking precedence over
    the key entered in the application sidebar.
    """
    openrouter_key = get_openrouter_api_key()
    if not openrouter_key:
        return False
    try:
        return not bool(get_gemini_api_key())
    except RuntimeError:
        return True

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
