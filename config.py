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
    # A configured key pool is an explicit manual setup. Start new sessions on
    # its first key, while keeping subsequent changes entirely user-selected.
    selected_index = st.session_state.get(
        "gemini_saved_key_index", 1 if configured_keys else 0
    )
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


GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")
GEMINI_FALLBACK_MODELS = ("gemini-3.5-flash", "gemini-2.5-flash", "gemini-2.5-pro")
GEMINI_MODELS = (GEMINI_MODEL, *GEMINI_FALLBACK_MODELS)

OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "google/gemini-3.1-pro-preview")
OPENROUTER_FALLBACK_MODEL = os.environ.get(
    "OPENROUTER_FALLBACK_MODEL", "google/gemini-3.7-flash"
)


def get_openrouter_api_key() -> str:
    """Retrieve an OpenRouter key from the active browser session or env."""
    session_key = st.session_state.get("openrouter_api_key", "").strip()
    if session_key:
        return session_key
    try:
        secret_key = st.secrets.get("OPENROUTER_API_KEY", "").strip()
        if secret_key:
            return secret_key
    except Exception:
        pass
    return os.environ.get("OPENROUTER_API_KEY", "").strip()


def use_openrouter() -> bool:
    """Use OpenRouter when an OpenRouter key has been configured and no direct Gemini key is available (or AI_PROVIDER is openrouter)."""
    if st.session_state.get("gemini_api_key"):
        return False
    if os.environ.get("AI_PROVIDER", "").lower() == "openrouter":
        return bool(get_openrouter_api_key())
    try:
        if get_gemini_api_key():
            return False
    except Exception:
        pass
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
