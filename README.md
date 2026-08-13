# Prescription Pad Cleaner

This Streamlit application creates two outputs from a scanned or photographed
prescription template:

- **Maximum-fidelity print output**: the real source page is detected,
  perspective-corrected, illumination-normalized, denoised, sharpened, and
  rendered as a 300-DPI PNG/PDF. Logos, photographs, watermarks, colors, and
  typography remain source-derived.
- **Structured editable output**: one vision request measures text, lines,
  colors, regions, and graphic bounds as validated JSON. A deterministic
  renderer rebuilds the page while original, unenhanced source crops are used
  for logos, seals, photos, and watermarks.

## Pipeline

```text
Input JPEG/PNG
  -> paper detection and perspective rectification
  -> full-resolution print restoration
  -> one structured document manifest (text + rules + artwork coordinates)
  -> deterministic editable HTML/CSS renderer
  -> original source graphic extraction from the rectified page
  -> visual similarity scoring and optional correction pass
  -> faithful PNG/PDF plus editable PNG/HTML
```

The print result does not depend on successful AI reconstruction. If the API is
unavailable or the editable layout cannot be measured, the source-faithful PNG
and PDF are still produced. For phone photos where page detection includes a
desk or surrounding objects, use the Manual page crop controls before running
the reconstruction.

## Setup

```powershell
cd prescription-cleaner-canva
python -m pip install -r requirements.txt
python -m playwright install chromium
```

Create `.streamlit/secrets.toml`:

```toml
GOOGLE_API_KEY = "your-key"
CANVA_ACCESS_TOKEN = ""
```

Run the project:

```powershell
python -m streamlit run app.py
```

Open `http://127.0.0.1:8501/`.

## Accuracy diagnostics

The app reports separate print and editable fidelity scores and displays a
difference heatmap. The editable browser document supports text editing and
movement/resizing of restored graphic assets.

Run the automated suite with:

```powershell
python -m pytest -q
```

The Chromium renderer is the supported Windows fallback when WeasyPrint's
native Pango/GObject libraries are unavailable.
