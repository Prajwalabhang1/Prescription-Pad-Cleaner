# Prescription Pad Cleaner & Canva Reconstruction System
## Complete Project Architectural Understanding & Technical Guide

---

## 1. Executive Summary & Core Mission

The **Prescription Pad Cleaner** is a specialized computer vision and multimodal AI application designed to convert scanned or mobile-photographed medical prescription pads (often skewed, wrinkled, unevenly lit, or containing handwritten text) into two distinct, high-value outputs:

1. **Maximum-Fidelity Restored Print Output (Deterministic Track)**:
   - Eliminates desk/floor surroundings, rectifies perspective distortion into a flat rectangular document, normalizes non-uniform camera illumination, denoises, and enhances contrast.
   - Preserves 100% of authentic source pixels for logos, hospital crests, stamps, watermarks, doctor signatures, and typography.
   - Exports high-resolution (300 DPI) print-ready **PNG** and **PDF** documents without risking AI hallucinations or altered medical information.

2. **Structured, Fully-Editable Document Reconstruction (Hybrid AI + CV Track)**:
   - Uses multimodal models (**Google Gemini 3.6 Flash / OpenRouter**) to recognize, transcribe, and lay out clean, editable HTML text for both Indic scripts (Hindi Devanagari, Telugu, Tamil) and English.
   - Solves the fundamental limitation of Generative AI (which cannot reliably redraw authentic logos, seals, or watermarks) through a **hybrid pipeline**: original visual assets are detected, cropped from the rectified source, cleaned of paper tint into transparent RGBA PNGs, and injected back into the editable HTML page as layered overlays.
   - Bundles an **in-browser, zero-account interactive editor** (click-to-edit text, drag-to-reposition images, scale, delete, download, or push directly to **Canva Connect**).

---

## 2. End-to-End System Architecture

```
                                  [ Input Image ]
                       (Mobile Camera Photo / Scan / Screenshot)
                                         │
                                         ▼
                   ┌───────────────────────────────────────────┐
                   │        1. Preprocessing & Rectification    │
                   │           (pipeline/preprocess.py)        │
                   │  - EXIF orientation transposition         │
                   │  - Contour / Edge / Paper quad detection  │
                   │  - Perspective warp to canonical page     │
                   │  - LAB illumination flattening & CLAHE   │
                   └─────────────────────┬─────────────────────┘
                                         │
                   ┌─────────────────────┴─────────────────────┐
                   ▼                                           ▼
   ┌───────────────────────────────┐           ┌───────────────────────────────┐
   │ Track A: Restored Source      │           │ Track B: Hybrid AI Rebuild    │
   │ - 300 DPI print restoration   │           │ - Dual ThreadPool execution:  │
   │ - Zero hallucination risk     │           │   * Text HTML: gemini_vision  │
   │ - Output: Faithful PNG & PDF  │           │   * Graphic BBox: doc_analyzer│
   └───────────────────────────────┘           └───────────────┬───────────────┘
                                                               │
                                         ┌─────────────────────┴─────────────────────┐
                                         ▼                                           ▼
                         ┌───────────────────────────────┐           ┌───────────────────────────────┐
                         │ Text Layout Generation        │           │ Asset Extraction & Guard      │
                         │ (pipeline/gemini_vision.py)   │           │ (asset_reconstruction.py)     │
                         │ - System prompt constraint    │           │ - Hough circle detection      │
                         │ - Indic + English font stacks │           │ - Transparent RGBA isolation  │
                         │ - Single-page container (.page│           │ - Watermark opacity capping   │
                         └───────────────┬───────────────┘           └───────────────┬───────────────┘
                                         │                                           │
                                         └─────────────────────┬─────────────────────┘
                                                               │
                                                               ▼
                                         ┌───────────────────────────────────────────┐
                                         │ 2. Hybrid Assembly & Injection            │
                                         │    (pipeline/template_renderer.py)        │
                                         │ - Merges text HTML with source graphics   │
                                         │ - Layering: Watermark (z=25), Text (z=20),│
                                         │   Artwork/Logo (z=40)                     │
                                         │ - Text auto-fitting script                │
                                         └─────────────────────┬─────────────────────┘
                                                               │
                                                               ▼
                                         ┌───────────────────────────────────────────┐
                                         │ 3. Deterministic HTML Rendering           │
                                         │    (pipeline/html_render.py)              │
                                         │ - Primary: Playwright (Chromium)          │
                                         │ - Fallback: WeasyPrint                    │
                                         │ - Single-page print contract enforcement  │
                                         └─────────────────────┬─────────────────────┘
                                                               │
                                                               ▼
                                         ┌───────────────────────────────────────────┐
                                         │ 4. Quality Scoring & Diagnostics          │
                                         │    (pipeline/visual_validation.py)        │
                                         │ - SSIM (Structural Similarity)            │
                                         │ - Edge F1 score (Canny contour match)     │
                                         │ - LAB DeltaE Color Similarity             │
                                         │ - Turbo-colormap difference heatmap       │
                                         └─────────────────────┬─────────────────────┘
                                                               │
                                                               ▼
                                         ┌───────────────────────────────────────────┐
                                         │ 5. Interactive Editing & Export           │
                                         │ - In-browser DOM editor (browser_editor.py│
                                         │ - Canva Connect API design push           │
                                         │ - Faithful & Editable PNG, PDF, HTML dl   │
                                         └───────────────────────────────────────────┘
```

---

## 3. Directory & File Inventory

```
prescription-cleaner-canva/
│
├── app.py                      # Main Streamlit UI & pipeline controller
├── config.py                   # Global configuration, API keys, and model parameters
├── Dockerfile                  # Production container definition (Python 3.11 + system libs)
├── docker-compose.yml          # Container deployment orchestrator
├── requirements.txt            # Python package dependencies
├── pytest.ini                  # Test configuration
│
├── pipeline/                   # Core business logic & processing engine
│   ├── __init__.py             # Module initializer
│   ├── asset_reconstruction.py # Crop, clean paper tint, and create transparent PNGs
│   ├── browser_editor.py       # Self-contained WYSIWYG editor engine (HTML/JS/CSS)
│   ├── canva_connect.py        # Canva Connect API integration (OAuth, upload, design)
│   ├── document_analyzer.py    # Vision AI prompt & parser for graphics manifest (JSON)
│   ├── document_manifest.py    # Dataclasses (NormalizedBox, DocumentElement, DocumentManifest)
│   ├── gemini_vision.py        # Vision AI engine for HTML/CSS text reconstruction
│   ├── html_render.py          # Chromium (Playwright) & WeasyPrint PDF/PNG renderer
│   ├── openrouter.py           # OpenRouter API client adapter for vision fallback
│   ├── page_geometry.py        # Physical dimension calculator (A4, portrait/landscape, mm/pt)
│   ├── pdf_export.py           # Fallback image-to-PDF utility
│   ├── preprocess.py           # CV pipeline: corner detection, perspective warp, lighting fix
│   ├── reconstruct.py          # Legacy canvas-drawing prototype (replaced by hybrid engine)
│   ├── refinement.py           # Optional second-pass visual manifest correction
│   ├── template_renderer.py    # Injects artwork layers & fit-text logic into HTML
│   ├── vision_ocr.py           # Legacy Google Cloud Vision OCR adapter
│   └── visual_validation.py    # Quantitative scoring (SSIM, Edge F1, Color) & heatmaps
│
├── tests/                      # Automated test suite
│   ├── test_accuracy_pipeline.py    # End-to-end integration tests
│   ├── test_browser_editor.py       # Editor HTML construction and publishing tests
│   ├── test_canva_connect.py        # Canva Connect API mock contract tests
│   ├── test_gemini_key_selection.py # Manual API key switching tests
│   ├── test_gemini_vision.py        # HTML validation, retries, and rate limit tests
│   ├── test_html_render.py          # Playwright & WeasyPrint rendering tests
│   ├── test_hybrid_reconstruction.py# Hybrid text-plus-source-artwork tests
│   ├── test_openrouter.py           # OpenRouter vision payload tests
│   ├── test_page_geometry.py        # Aspect ratio and orientation tests
│   ├── test_preprocess.py           # Image rectification and resolution cap tests
│   └── test_provider_selection.py   # Key priority & provider selection tests
│
└── .streamlit/
    └── secrets.toml            # Local credentials (API keys, Canva tokens)
```

---

## 4. Deep-Dive into Pipeline Components

### 4.1 Configuration & Key Management (`config.py`)
- **Multi-key rotation & fallback**:
  Supports `GEMINI_API_KEYS` (a list of keys in `secrets.toml`). If a key hits a 429 quota exhaustion, users can manually switch keys from the sidebar without modifying files or restarting the app.
- **Provider Priority**:
  1. Browser session manual key override (`gemini_api_key` in session state).
  2. Selected saved key from `GEMINI_API_KEYS`.
  3. System environment variable `GEMINI_API_KEY` or `GOOGLE_API_KEY`.
  4. Local `.streamlit/secrets.toml`.
  5. OpenRouter fallback if configured and no Gemini key is available.
- **Models**:
  - Primary: `gemini-3.6-flash` (superior spatial reasoning, supports `ThinkingConfig`).
  - Fallbacks: `gemini-3.5-flash`, `gemini-2.5-flash`, `gemini-2.5-pro`.
  - OpenRouter default: `google/gemini-2.5-flash`.

---

### 4.2 Image Preprocessing & Computer Vision (`pipeline/preprocess.py`)
Prescriptions uploaded from mobile phones have 4 major distortions: rotation, background intrusion (desks/shadows), perspective keystone skew, and non-uniform lighting.

1. **Orientation**: `ImageOps.exif_transpose` corrects camera sensor orientation metadata.
2. **Page Corner Detection (`_detect_page`)**:
   - Runs multiscale segmentation: Otsu thresholding + adaptive thresholds (130, 150, 170, 190) on lightness (LAB space) and Canny edge detection.
   - Finds convex quadrilaterals with `cv2.approxPolyDP` and `cv2.minAreaRect`.
   - Scores candidates via `_candidate_confidence` measuring interior area, border margins, and foreground-to-background contrast.
3. **Perspective Warp (`_warp_page`)**:
   - Uses `cv2.getPerspectiveTransform` and `cv2.warpPerspective` with cubic interpolation to transform the quadrilateral into a flat rectangular canonical image.
4. **Illumination Normalization (`_normalize_illumination`)**:
   - Converts to LAB color space.
   - Extracts lightness $L$, estimates the background illumination field via large-kernel Gaussian blur ($\sigma \approx \max(\text{dim})/28$), and divides $L$ by the estimated background field.
   - This removes shadows, flash glare, and yellow paper tints while preserving colored ink, blue logos, and red doctor titles.
5. **Print Restoration vs Analysis Sizing**:
   - `restored`: Scaled to 3508 px (A4 @ 300 DPI) with bilateral filtering and unsharp masking.
   - `analysis`: Capped at 2048 px (`MAX_ANALYSIS_DIMENSION`) for fast, token-efficient AI vision inference without downscaling fine font details.

---

### 4.3 Multimodal Layout & Text Reconstruction (`pipeline/gemini_vision.py`)
Generates single-page HTML/CSS that mirrors the exact layout of the prescription.

- **Strict Prompt Engineering**:
  - Prohibits markdown prose, multi-page documents, and inline `style="..."` attributes (mandates centralized `<style>` rules to conserve tokens).
  - Explicit font requirements: `Noto Sans Devanagari` / `Noto Sans Telugu` for Indic scripts, serif (`Playfair Display`, `EB Garamond`) or sans-serif (`Inter`, `Roboto`) for English.
  - Prohibits redrawing logos, stamps, or watermarks: instructs model to insert empty `<div class="logo-spacer">` to hold whitespace so overlays will not obscure text.
- **Strict Document Boundary Validation (`validate_reconstruction_html`)**:
  - Rejects truncated or malformed responses missing `<!DOCTYPE html>`, `<html>`, `<head>`, `</head>`, `<body>`, `</body>`, or `</html>`.
  - Verifies that `<style>` and `<script>` tags are strictly balanced.
  - Rejects visually empty documents.
- **Resilience & Rate-Limiting**:
  - Detects Gemini `429 RESOURCE_EXHAUSTED` and extracts exact `retryDelay` from the error payload.
  - Retries transient network failures (500, 502, 503, 504, `DEADLINE_EXCEEDED`) up to 3 times, automatically compressing retry payloads to JPEG at 1600 px to reduce upload latency.
  - Automatic model failover from `gemini-3.6-flash` to `gemini-3.5-flash` if Google services report high demand.

---

### 4.4 Graphics Localization & Manifest (`pipeline/document_analyzer.py` & `pipeline/document_manifest.py`)
Rather than relying on the model to author complex SVG code:
- `analyze_graphics` prompts the vision model to return a **pure JSON manifest** specifying normalized bounding boxes `[x, y, width, height]` (0.0 to 1.0) and semantic roles:
  - `logo` / `medical_icon` / `emblem`: Clinic crests, symbols, caduceus.
  - `watermark_photo`: Faint background portrait or baby image.
  - `watermark_seal`: Circular faint clinic stamp in the writable body.
  - `signature` / `photo` / `seal`.
- **Classification Rules**:
  - Artwork in header region ($y < 0.25$) is forced to `logo` or `medical_icon` (never watermark).
  - Watermarks are restricted to the prescription body ($y \ge 0.16$).
  - Full-page coverage bounding boxes ($[0, 0, 1, 1]$) are rejected by `sanitize_graphics`.

---

### 4.5 Asset Extraction & Background Cleaning (`pipeline/asset_reconstruction.py`)
Converts rectangular raw image crops into clean, transparent, print-quality graphic assets.

1. **Complete Bounds Recovery (`complete_graphic_bounds`)**:
   - AI vision bounding boxes often truncate logos (e.g. cutting through a circular crest or doctor nameplate).
   - `_find_logo_circle` runs OpenCV Hough Circle Transform (`cv2.HoughCircles`) in the top-left quadrant ($x < 0.23, y < 0.17$) to detect the true circular emblem boundary and snap the bounding box to it.
   - `_find_watermark_circle` locates circular watermark stamps in the body area.
2. **Paper Tint Removal & Alpha Generation (`_clean_paper_background`)**:
   - Estimates local paper background lightness and saturation using adaptive Gaussian blur.
   - Computes ink difference relative to paper background:
     $$\text{ink\_luma\_diff} = \max(\text{paper\_bg\_gray} - \text{gray}, 0)$$
   - Generates soft alpha gradients with feathered edges ($3\%$ boundary ramp) and Gaussian smoothing ($\sigma = 0.45$).
   - Normalizes and white-balances RGB channels against local paper background so yellowed/grayish paper becomes pure transparent white, while retaining vivid colored logo inks.
3. **Photo Watermark Preservation (`_transparent_photo_watermark`)**:
   - Applies soft alpha threshold capped at 0.28 opacity so background baby photos or hospital watermarks do not overpower handwritten or printed clinical text.

---

### 4.6 Template Assembly & Text Fitting (`pipeline/template_renderer.py`)
Combines the generated HTML text layout with the extracted source graphics:

- **CSS Layering Contract (`inject_source_graphics`)**:
  - Injects isolated layers into `<div class="page">`:
    - `.source-watermark-layer`: `z-index: 25` (positioned beneath text but above body backgrounds).
    - Text and content: `position: relative; z-index: 20`.
    - `.source-artwork-layer`: `z-index: 40` (logos and crests sit on top of headers).
- **Auto-Fit Engine (`manifest-text-fit`)**:
  - Injects a zero-dependency client-side script that executes on font load (`document.fonts.ready`).
  - Measures `scrollWidth` vs `clientWidth`. If a long translated doctor title or clinic address overflows its container, it iteratively steps down font size ($0.94\times$ per step) and applies CSS `scale()` transforms to guarantee text never clips or wraps awkwardly.

---

### 4.7 Rendering Engine (`pipeline/html_render.py`)
Converts HTML into high-resolution 300 DPI raster PNG and vector PDF.

- **Primary: Playwright Chromium**:
  - Headless Chromium provides pixel-perfect OpenType font shaping, CSS Flexbox/Grid support, and font rendering (crucial for Indic Devanagari ligatures).
  - Sets viewport dynamically matching document millimeter dimensions converted to pixels at target DPI:
    $$\text{viewport\_px} = \text{round}\left(\frac{\text{dim\_mm}}{25.4} \times \text{DPI}\right)$$
  - Emulates print media (`page_obj.emulate_media(media="print")`) and exports native PDF with zero margins.
  - Windows Event Loop Support: automatically initializes `WindowsProactorEventLoopPolicy` to prevent `NotImplementedError` on Windows systems.
  - Protocol Error Fallback: if Chromium's PDF driver encounters printer buffer errors, it takes a full-page 300-DPI screenshot and wraps it into a PDF via PyMuPDF.
- **Fallback: WeasyPrint**:
  - Native Python/C Cairo+Pango HTML-to-PDF library. Used if Playwright Chromium is not installed or available.

---

### 4.8 Quality Validation & Difference Heatmaps (`pipeline/visual_validation.py`)
Measures reconstruction fidelity against the original rectified document:

1. **SSIM (Structural Similarity Index)**: Measures structural pattern matching between normalized grayscale source and reconstruction:
   $$\text{SSIM}(x, y) = \frac{(2\mu_x\mu_y + c_1)(2\sigma_{xy} + c_2)}{(\mu_x^2 + \mu_y^2 + c_1)(\sigma_x^2 + \sigma_y^2 + c_2)}$$
2. **Edge F1 Score**: Runs Canny edge detection on both images, dilates with a $3\times 3$ kernel, and computes the harmonic mean of precision and recall between edge boundaries.
3. **Color Similarity**: Computes Euclidean distance in CIELAB color space ($\Delta E$) and maps it through an exponential decay function:
   $$\text{Color} = \exp\left(-\frac{\text{mean}(\Delta E)}{18.0}\right)$$
4. **Overall Score**:
   $$\text{Overall} = 0.55 \times \text{SSIM} + 0.35 \times \text{Edge F1} + 0.10 \times \text{Color}$$
5. **Difference Heatmap**: Generates a Turbo colormap visual diff overlay highlighting misaligned text lines, missing borders, or shifted logo locations.

---

### 4.9 In-Browser WYSIWYG Editor (`pipeline/browser_editor.py`)
- Self-contained HTML document requiring **no accounts, external databases, or third-party servers**.
- Injects a floating dark-mode toolbar (`#prescription-editor-toolbar`):
  - **Click-to-edit**: Sets `contentEditable="true"` on headings, paragraphs, and table cells.
  - **Drag-and-drop**: Uses HTML5 PointerEvents with `setPointerCapture` to allow clicking and dragging logos or watermark images anywhere across the canvas.
  - **Controls**: `+ Text`, `Smaller (-)`, `Larger (+)`, arrow nudges (`←`, `↑`, `↓`, `→`), `Delete`, `Print`, and `Download edited HTML`.
- Published locally to `/app/static/editor/prescription-<hash>.html` and accessible via `data:text/html;base64,...` URI.

---

### 4.10 Canva Connect Integration (`pipeline/canva_connect.py`)
- Connects to Canva Connect REST API (`https://api.canva.com/rest/v1`).
- `upload_asset`: Sends reconstructed PNG as an octet-stream with base64 metadata, polling `/asset-uploads/{job_id}` until completion.
- `create_design`: Creates a new custom design matching the exact aspect ratio and dimensions of the prescription pad and returns a direct `edit_url` opening Canva's editor.

---

## 5. Execution Flow Trace (Step-by-Step)

When a user uploads a prescription in `app.py`:

```
User uploads "prescription.jpg"
  │
  ├── 1. app.py computes SHA256 file key. Checks session state cache.
  │
  ├── 2. Preprocessing (preprocess_document)
  │      ├── Decodes EXIF & detects page boundaries (Otsu + Canny quads)
  │      ├── Warps perspective to canonical flat page
  │      ├── Normalizes illumination (LAB division)
  │      └── Prepares canonical (raw), restored (3508px print), and analysis (2048px)
  │
  ├── 3. Maximum-Fidelity Track (build_fidelity_html -> render_html)
  │      ├── Renders restored image into 300 DPI PNG & PDF
  │      └── Computes baseline Print Fidelity score (typically 99.5%+)
  │
  ├── 4. AI Parallel Reconstruction (ThreadPoolExecutor, max_workers=2)
  │      ├── Thread 1: generate_clean_html(analysis_bytes) -> Gemini Vision (HTML/CSS)
  │      └── Thread 2: analyze_graphics(analysis_bytes)    -> Gemini Vision (JSON bbox manifest)
  │
  ├── 5. Hybrid Asset Fusion
  │      ├── sanitize_graphics drops invalid bounding boxes (e.g. text mislabeled as watermark)
  │      ├── complete_graphic_bounds snaps boxes to Hough circles (logos & seals)
  │      ├── reconstruct_assets crops canonical page, isolates alpha, removes paper tint
  │      └── inject_source_graphics inserts layered overlays into clean HTML
  │
  ├── 6. Render & Validation
  │      ├── Chromium renders candidate HTML to 300 DPI image & PDF
  │      ├── compare_images evaluates SSIM, Edge F1, and Color score
  │      └── difference_heatmap generates visual diagnostic map
  │
  └── 7. Output Display & Export
         ├── Side-by-side comparison: Rectified Original vs Restored Print
         ├── Score metrics (Print Fidelity %, Editable Fidelity %, Dimensions)
         ├── Interactive Browser Editor tab & Canva export button
         └── 6 Direct Downloads: Faithful PNG/PDF/HTML & Editable PNG/PDF/HTML
```

---

## 6. Identified Deficiencies, Bugs, and Recommended Fixes

| Component | Root Cause | Impact | Fix |
| :--- | :--- | :--- | :--- |
| **`pipeline/refinement.py`** | Dangling, unclosed API call fragment at lines 36–45 | Cascading Python `SyntaxError` preventing compilation | Delete lines 36–45, restoring single `client.models.generate_content` call |
| **`pipeline/asset_reconstruction.py`** | `_find_logo_circle` missing; fixed 0.008 expansion used | AI bounding boxes that truncate circular logos fail to recover | Reintroduce `_find_logo_circle` using OpenCV Hough Circles |
| **`pipeline/asset_reconstruction.py`** | Aggressive `text_suppression` in `_clean_paper_background` | Watermarks with contrast difference > 40 are zeroed out (alpha=0) | Remove hard subtraction or soften threshold |
| **`pipeline/asset_reconstruction.py`** | `_asset_with_alpha` does not route photo watermarks | Photo watermarks receive generic watermark opacity (0.38 vs 0.28) | Check `_watermark_kind(role) == "photo"` and dispatch to `_transparent_photo_watermark` |
| **`pipeline/document_analyzer.py`** | Contract string missing from `GRAPHICS_PROMPT` | Model may enclose body text inside artwork boxes; unit test fails | Restore `"Do not include any printed letters, numbers, bullet lists, or prescription body fields in an artwork bbox."` |
| **`pipeline/preprocess.py`** | `MAX_ANALYSIS_DIMENSION` changed to 1400 | Reduces OCR sharpness for small Devanagari text; unit test fails | Restore `MAX_ANALYSIS_DIMENSION = 2048` |
| **`config.py`** | `OPENROUTER_MODEL` set to `gemini-2.5-pro` | Mismatches unit test expecting `google/gemini-2.5-flash` | Restore default to `"google/gemini-2.5-flash"` |
| **`pipeline/reconstruct.py` & `vision_ocr.py`** | Imports symbols deleted from `config.py` (`TEMPLATE`, `VISION_ENDPOINT`) | `ImportError` on direct import | Provide fallback definitions or deprecate legacy modules |
| **`Dockerfile`** | Missing `playwright install chromium` | Playwright fails inside Docker container | Add `RUN python -m playwright install --with-deps chromium` |

---

## 7. Developer & Operations Quickstart

### 7.1 Running Locally
```powershell
# 1. Navigate to repository root
cd prescription-cleaner-canva

# 2. Install dependencies
python -m pip install -r requirements.txt

# 3. Install Playwright browser engine
python -m playwright install chromium

# 4. Configure secrets in .streamlit/secrets.toml
# GOOGLE_API_KEY = "your-api-key"
# GEMINI_API_KEYS = ["key1", "key2"]
# CANVA_ACCESS_TOKEN = "" # optional

# 5. Run the Streamlit application
python -m streamlit run app.py
```

### 7.2 Running Test Suite
```powershell
python -m pytest -v
```

---
*Document compiled for developers, code reviewers, and system architects.*
