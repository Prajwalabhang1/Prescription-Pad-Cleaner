"""Streamlit application for faithful and editable prescription reconstruction."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import base64
from dataclasses import replace
import hashlib
import os

# Keep the original macOS native-library fallback for WeasyPrint users.
if "DYLD_FALLBACK_LIBRARY_PATH" not in os.environ:
    os.environ["DYLD_FALLBACK_LIBRARY_PATH"] = "/opt/homebrew/lib"

import streamlit as st

from pipeline.asset_reconstruction import (
    complete_graphic_bounds,
    reconstruct_assets,
    sanitize_graphics,
)
from pipeline.browser_editor import build_editor_html, publish_editor_html
from pipeline.document_analyzer import analyze_graphics
from pipeline.document_manifest import DocumentManifest, NormalizedBox
from pipeline.gemini_vision import GeminiRateLimitError, generate_clean_html
from pipeline.html_render import prepare_print_html, render_html
from pipeline.page_geometry import PageGeometry
from pipeline.preprocess import pil_to_bytes, preprocess_document
from pipeline.template_renderer import build_fidelity_html, inject_source_graphics
from pipeline.visual_validation import compare_images, difference_heatmap


st.set_page_config(
    page_title="Prescription Pad Cleaner",
    page_icon="P",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    .stApp { font-family: 'Inter', sans-serif; }
    .main-header {
        background: #172033;
        border: 1px solid #35425a;
        border-left: 5px solid #ef476f;
        padding: 1.35rem 1.6rem;
        border-radius: 8px;
        margin-bottom: 1.25rem;
    }
    .main-header h1 { color: #f8fafc; font-size: 1.8rem; margin: 0 0 0.3rem; }
    .main-header p { color: #b9c3d2; font-size: 0.92rem; margin: 0; }
    .pipeline-step {
        border-left: 3px solid #3b82f6;
        padding: 0.55rem 0.75rem;
        margin: 0.35rem 0;
        color: #cbd5e1;
        font-size: 0.85rem;
    }
    div[data-testid="stFileUploader"] { border: 1px dashed #64748b; border-radius: 8px; }
    </style>
    <div class="main-header">
      <h1>Prescription Pad Cleaner</h1>
      <p>Rectify the source, preserve real artwork, measure reconstruction accuracy, and export print-ready files.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

PIPELINE_KEYS = [
    "pipeline_done",
    "canonical_source",
    "restored_source",
    "fidelity_html",
    "result_img",
    "png_bytes",
    "pdf_bytes",
    "editable_html",
    "editable_img",
    "editable_png",
    "editor_html",
    "editor_url",
    "heatmap",
    "fidelity_score",
    "editable_score",
    "analysis_error",
    "page_detected",
    "page_confidence",
    "page_detection_method",
    "artwork_assets",
    "artwork_manifest",
    "artwork_base_manifest",
    "editable_base_html",
    "artwork_warnings",
    "_file_key",
]


def _clear_pipeline() -> None:
    for state_key in PIPELINE_KEYS:
        st.session_state.pop(state_key, None)


def _set_session_api_key() -> None:
    """Use the sidebar key for this browser session without persisting it to disk."""
    key = st.session_state.get("gemini_api_key_input", "").strip()
    previous = st.session_state.get("gemini_api_key", "")
    if key == previous:
        return
    if key:
        st.session_state["gemini_api_key"] = key
    else:
        st.session_state.pop("gemini_api_key", None)
    _clear_pipeline()


def _data_uri_bytes(data_uri: str) -> bytes:
    return base64.b64decode(data_uri.split(",", 1)[1])


def _apply_artwork_adjustments() -> None:
    """Recompose source artwork without spending another vision-model request."""
    state = st.session_state
    base_manifest: DocumentManifest = state["artwork_base_manifest"]
    elements = []
    for element in base_manifest.elements:
        key = f"artwork-{element.id}"
        offset_x = state.get(f"{key}-x", 0.0) / 100
        offset_y = state.get(f"{key}-y", 0.0) / 100
        scale = state.get(f"{key}-scale", 1.0)
        box = element.box
        adjusted = NormalizedBox.from_value(
            [
                box.x + offset_x - box.width * (scale - 1) / 2,
                box.y + offset_y - box.height * (scale - 1) / 2,
                box.width * scale,
                box.height * scale,
            ]
        )
        opacity = state.get(f"{key}-opacity", element.opacity)
        elements.append(replace(element, box=adjusted, opacity=opacity))
    graphics = replace(base_manifest, elements=tuple(elements))
    page = PageGeometry.from_pixels(*state.canonical_source.size)
    html = inject_source_graphics(
        state.editable_base_html, graphics, state.artwork_assets, page
    )
    image, _ = render_html(html, state.get("render_dpi", 300), page)
    state.editable_html = html
    state.editable_img = image
    state.editable_png = pil_to_bytes(image)
    state.editable_score = compare_images(state.restored_source, image)
    state.editor_html = build_editor_html(html)
    state.editor_url = publish_editor_html(state.editor_html)
    state.heatmap = difference_heatmap(state.restored_source, image)
    state.artwork_manifest = graphics


with st.sidebar:
    st.markdown("### Settings")
    st.text_input(
        "Direct Gemini API key (optional)",
        type="password",
        key="gemini_api_key_input",
        placeholder="Paste API key",
        help="Used only in this browser session. It is not saved to a project file.",
        on_change=_set_session_api_key,
    )
    if st.session_state.get("gemini_api_key"):
        st.caption("Session API key active")
    dpi = st.select_slider("Render DPI", options=[150, 200, 250, 300], value=300)
    use_manual_crop = st.toggle(
        "Manual page crop",
        key="use_manual_crop",
        help="Use this when the page boundary includes desk, floor, or other camera background.",
    )
    manual_crop = None
    if use_manual_crop:
        st.caption("Remove camera surroundings before reconstruction.")
        crop_left = st.slider("Crop left", 0.0, 0.45, 0.0, 0.01)
        crop_top = st.slider("Crop top", 0.0, 0.45, 0.0, 0.01)
        crop_right = st.slider("Crop right", 0.55, 1.0, 1.0, 0.01)
        crop_bottom = st.slider("Crop bottom", 0.55, 1.0, 1.0, 0.01)
        if crop_right - crop_left < 0.25 or crop_bottom - crop_top < 0.25:
            st.error("Manual crop must retain at least one quarter of the page.")
        else:
            manual_crop = (crop_left, crop_top, crop_right, crop_bottom)
    st.markdown("---")
    st.markdown(
        """
        **Pipeline**
        <div class="pipeline-step">1. Detect, rectify, and restore the page</div>
        <div class="pipeline-step">2. Generate clean editable text and layout</div>
        <div class="pipeline-step">3. Restore original logo and watermark layers</div>
        <div class="pipeline-step">4. Export faithful and editable versions</div>
        """,
        unsafe_allow_html=True,
    )

uploaded = st.file_uploader(
    "Upload a scanned prescription pad image",
    type=["png", "jpg", "jpeg"],
    help="Accepts phone photographs, scans, and screenshots.",
)


if uploaded:
    image_bytes = uploaded.getvalue()
    digest = hashlib.sha256(image_bytes).hexdigest()
    crop_key = "auto" if manual_crop is None else ":".join(f"{value:.2f}" for value in manual_crop)
    file_key = f"{digest}_{dpi}_{crop_key}_hybrid-text-v6-multitemplate"
    if st.session_state.get("_file_key") != file_key:
        _clear_pipeline()
        st.session_state["_file_key"] = file_key

    if not st.session_state.get("pipeline_done"):
        st.image(uploaded, caption="Uploaded image", width=400)
        if not st.button("Start processing", type="primary", use_container_width=True):
            st.stop()

        with st.spinner("Detecting and rectifying the prescription page..."):
            try:
                processed = preprocess_document(image_bytes, manual_crop=manual_crop)
                source_page = PageGeometry.from_pixels(*processed.page_size)
            except Exception as error:
                st.error(f"The uploaded image could not be prepared: {error}")
                st.stop()

        with st.spinner("Creating maximum-fidelity print artwork..."):
            try:
                fidelity_html = build_fidelity_html(processed.restored, source_page)
                result_img, pdf_bytes = render_html(fidelity_html, dpi, source_page)
                fidelity_score = compare_images(processed.restored, result_img)
            except Exception as error:
                st.error(f"High-fidelity rendering failed: {error}")
                st.stop()

        analysis_error = ""
        editable_html = fidelity_html
        editable_img = result_img
        editable_score = fidelity_score

        with st.spinner("Generating editable text and restoring source graphics in parallel..."):
            try:
                analysis_bytes = pil_to_bytes(processed.analysis)
                # These Gemini requests inspect the same page but have no dependency
                # on each other. Running them together shortens the wait without
                # changing the model, prompt, source pixels, or output settings.
                with ThreadPoolExecutor(max_workers=2) as executor:
                    text_future = executor.submit(
                        generate_clean_html, analysis_bytes, source_page
                    )
                    graphics_future = executor.submit(
                        analyze_graphics, analysis_bytes, source_page
                    )
                    text_html = text_future.result()
                    graphics, artwork_warnings = sanitize_graphics(graphics_future.result())
                    graphics = complete_graphic_bounds(processed.analysis, graphics)
                text_html = prepare_print_html(text_html, source_page)

                assets = reconstruct_assets(processed.restored, graphics)
                candidate_html = inject_source_graphics(
                    text_html, graphics, assets, source_page
                )
                candidate_img, _candidate_pdf = render_html(candidate_html, dpi, source_page)
                candidate_score = compare_images(processed.restored, candidate_img)
                editable_html = candidate_html
                editable_img = candidate_img
                editable_score = candidate_score
            except GeminiRateLimitError as error:
                analysis_error = (
                    "Editable text reconstruction is waiting on the Gemini API quota. "
                    f"{error} The faithful restored page remains available meanwhile."
                )
            except Exception as error:
                analysis_error = (
                    "Editable text reconstruction was unavailable, so the editor uses "
                    f"the faithful restored page. Details: {error}"
                )

        try:
            editor_html = build_editor_html(editable_html)
            editor_url = publish_editor_html(editor_html)
            heatmap = difference_heatmap(processed.restored, editable_img)
        except Exception as error:
            st.error(f"The output editor could not be prepared: {error}")
            st.stop()

        with open("latest_fidelity.html", "w", encoding="utf-8") as output:
            output.write(fidelity_html)
        with open("latest_output.html", "w", encoding="utf-8") as output:
            output.write(editable_html)

        st.session_state.update(
            pipeline_done=True,
            canonical_source=processed.canonical,
            restored_source=processed.restored,
            fidelity_html=fidelity_html,
            result_img=result_img,
            png_bytes=pil_to_bytes(result_img),
            pdf_bytes=pdf_bytes,
            editable_html=editable_html,
            editable_img=editable_img,
            editable_png=pil_to_bytes(editable_img),
            editor_html=editor_html,
            editor_url=editor_url,
            heatmap=heatmap,
            fidelity_score=fidelity_score,
            editable_score=editable_score,
            analysis_error=analysis_error,
            page_detected=processed.page_corners is not None,
            page_confidence=processed.page_confidence,
            page_detection_method=processed.page_detection_method,
            artwork_assets=assets if not analysis_error else {},
            artwork_warnings=artwork_warnings if not analysis_error else (),
            artwork_manifest=graphics if not analysis_error else DocumentManifest(),
            artwork_base_manifest=graphics if not analysis_error else DocumentManifest(),
            editable_base_html=text_html if not analysis_error else fidelity_html,
            render_dpi=dpi,
        )

    state = st.session_state
    st.markdown("---")
    source_column, output_column = st.columns(2)
    with source_column:
        st.markdown("#### Original (rectified)")
        st.image(state.canonical_source, use_container_width=True)
    with output_column:
        st.markdown("#### Maximum-fidelity clean output")
        st.image(state.result_img, use_container_width=True)

    metric1, metric2, metric3 = st.columns(3)
    metric1.metric("Print fidelity", f"{state.fidelity_score.overall * 100:.1f}%")
    metric2.metric("Editable fidelity", f"{state.editable_score.overall * 100:.1f}%")
    metric3.metric("Render resolution", f"{state.result_img.width} x {state.result_img.height}")
    if not state.page_detected:
        st.warning(
            "A confident paper boundary was not found. The full image was retained "
            "to avoid accidentally cropping prescription content."
        )
    elif state.page_confidence < 0.72:
        st.warning(
            "Page boundary needs review. Enable Manual page crop in the sidebar to "
            "remove desk, floor, shadows, or camera surroundings before processing again."
        )
    st.caption(
        f"Page detection: {state.page_detection_method} "
        f"({state.page_confidence * 100:.0f}% confidence)"
    )
    for warning in state.artwork_warnings:
        st.warning(warning)
    if state.analysis_error:
        st.warning(state.analysis_error)

    if state.artwork_assets:
        with st.expander("Extracted artwork review", expanded=False):
            preview_columns = st.columns(min(3, len(state.artwork_assets)))
            roles = {
                element.id: element.role for element in state.artwork_base_manifest.elements
            }
            for column, (asset_id, asset) in zip(preview_columns, state.artwork_assets.items()):
                with column:
                    st.image(
                        _data_uri_bytes(asset),
                        caption=f"{asset_id} ({roles.get(asset_id, 'artwork')})",
                        use_container_width=True,
                    )
            st.markdown("#### Placement adjustments")
            for element in state.artwork_base_manifest.elements:
                key = f"artwork-{element.id}"
                controls = st.columns(4)
                with controls[0]:
                    st.number_input("X offset (%)", -20.0, 20.0, 0.0, 0.1, key=f"{key}-x")
                with controls[1]:
                    st.number_input("Y offset (%)", -20.0, 20.0, 0.0, 0.1, key=f"{key}-y")
                with controls[2]:
                    st.number_input("Scale", 0.5, 1.6, 1.0, 0.01, key=f"{key}-scale")
                with controls[3]:
                    st.number_input(
                        "Opacity", 0.05, 1.0, float(element.opacity), 0.01, key=f"{key}-opacity"
                    )
            if st.button("Apply artwork adjustments", use_container_width=True):
                _apply_artwork_adjustments()
                st.rerun()

    editable_tab, difference_tab = st.tabs(
        ["Editable reconstruction", "Difference heatmap"]
    )
    with editable_tab:
        st.image(state.editable_img, use_container_width=True)
    with difference_tab:
        st.image(state.heatmap, use_container_width=True)

    st.markdown("---")
    st.markdown("### Download outputs")
    download1, download2, download3 = st.columns(3)
    with download1:
        st.download_button(
            "Download faithful PNG",
            state.png_bytes,
            "clean_pad_faithful.png",
            "image/png",
            use_container_width=True,
        )
    with download2:
        st.download_button(
            "Download faithful PDF",
            state.pdf_bytes,
            "clean_pad_faithful.pdf",
            "application/pdf",
            use_container_width=True,
        )
    with download3:
        st.download_button(
            "Download editable PNG",
            state.editable_png,
            "clean_pad_editable.png",
            "image/png",
            use_container_width=True,
        )

    download4, download5, download6 = st.columns(3)
    with download4:
        st.download_button(
            "Download faithful HTML",
            state.fidelity_html,
            "clean_pad_faithful.html",
            "text/html",
            use_container_width=True,
        )
    with download5:
        st.download_button(
            "Download editable HTML",
            state.editor_html,
            "clean_pad_editable.html",
            "text/html",
            use_container_width=True,
        )
    with download6:
        st.link_button(
            "Open browser editor",
            state.editor_url,
            type="primary",
            use_container_width=True,
        )

    if st.button("Process another image"):
        _clear_pipeline()
        st.rerun()
else:
    _clear_pipeline()
    st.markdown(
        """
        <div style="text-align:center;padding:3.5rem 2rem;color:#94a3b8;">
          <h2 style="color:#e2e8f0;font-size:1.45rem;">Upload a prescription template to begin</h2>
          <p>The print result preserves real source graphics; the editable result is measured separately.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
