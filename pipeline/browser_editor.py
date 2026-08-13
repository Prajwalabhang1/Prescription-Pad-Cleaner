"""Build and publish a no-account, browser-based reconstruction editor.

The editor is intentionally self-contained: the reconstructed HTML, editor
controls, and JavaScript live in one file.  It can therefore be opened from
the app's static URL or downloaded and opened later without a Canva account,
API key, or network connection.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EDITOR_DIRECTORY = PROJECT_ROOT / "static" / "editor"
EDITOR_URL_PREFIX = "/app/static/editor"
EDITOR_MARKER = "prescription-browser-editor"


class BrowserEditorError(RuntimeError):
    """Raised when a reconstructed document cannot be turned into an editor."""


EDITOR_STYLE = f"""
<style id="{EDITOR_MARKER}-style">
#prescription-editor-toolbar {{
  position: fixed;
  right: 16px;
  bottom: 16px;
  z-index: 2147483647;
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  align-items: center;
  max-width: min(560px, calc(100vw - 32px));
  padding: 10px;
  border: 1px solid #cbd5e1;
  border-radius: 12px;
  background: rgba(15, 23, 42, 0.96);
  box-shadow: 0 12px 32px rgba(15, 23, 42, 0.28);
  color: #f8fafc;
  font: 600 13px/1.2 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}}

#prescription-editor-toolbar .editor-title {{
  width: 100%;
  color: #bfdbfe;
  font-size: 12px;
}}

#prescription-editor-toolbar button {{
  appearance: none;
  min-height: 30px;
  padding: 6px 9px;
  border: 1px solid #475569;
  border-radius: 7px;
  background: #1e293b;
  color: #f8fafc;
  font: inherit;
  cursor: pointer;
}}

#prescription-editor-toolbar button:hover {{
  background: #334155;
}}

#prescription-editor-toolbar button.editor-primary {{
  border-color: #38bdf8;
  background: #0369a1;
}}

#prescription-editor-toolbar #editor-status {{
  margin-left: 2px;
  color: #cbd5e1;
  font-size: 11px;
}}

[data-editor-text="true"]:hover {{
  outline: 1px dashed rgba(14, 116, 144, 0.9);
  outline-offset: 2px;
}}

[data-editor-object="true"] {{
  touch-action: none;
  cursor: grab;
}}

[data-editor-object="true"]:active {{
  cursor: grabbing;
}}

.editor-selected {{
  outline: 2px solid #0ea5e9 !important;
  outline-offset: 3px !important;
}}

@media print {{
  #prescription-editor-toolbar {{ display: none !important; }}
  [data-editor-text="true"]:hover,
  .editor-selected {{ outline: none !important; }}
}}
</style>
"""


EDITOR_UI = """
<div id="prescription-editor-toolbar" role="toolbar" aria-label="Prescription editor">
  <div class="editor-title">Click text to edit · Drag images · Select an item, then use controls</div>
  <button type="button" data-editor-action="add-text">+ Text</button>
  <button type="button" data-editor-action="smaller" title="Smaller text or object">−</button>
  <button type="button" data-editor-action="larger" title="Larger text or object">+</button>
  <button type="button" data-editor-action="left" title="Move selected item left">←</button>
  <button type="button" data-editor-action="up" title="Move selected item up">↑</button>
  <button type="button" data-editor-action="down" title="Move selected item down">↓</button>
  <button type="button" data-editor-action="right" title="Move selected item right">→</button>
  <button type="button" data-editor-action="delete" title="Delete selected item">Delete</button>
  <button type="button" data-editor-action="print">Print</button>
  <button type="button" class="editor-primary" data-editor-action="download">Download edited HTML</button>
  <span id="editor-status" aria-live="polite">Ready</span>
</div>
<script id="prescription-browser-editor">
(() => {
  const toolbar = document.getElementById("prescription-editor-toolbar");
  const page = document.querySelector("body > .page, body > #prescription-page") || document.body;
  const status = document.getElementById("editor-status");
  const textSelector = "h1,h2,h3,h4,h5,h6,p,span,div,li,td,th,label,a,strong,em,b,i";
  let selected = null;
  let drag = null;

  const setStatus = (message) => { status.textContent = message; };

  const markText = (element) => {
    if (element.closest("#prescription-editor-toolbar")) return;
    if (element.children.length === 0 && element.textContent.trim()) {
      element.dataset.editorText = "true";
      element.dataset.editorNode = "true";
      element.contentEditable = "true";
      element.spellcheck = false;
    }
  };

  page.querySelectorAll(textSelector).forEach(markText);
  page.querySelectorAll("img, svg, canvas").forEach((element) => {
    element.dataset.editorObject = "true";
    element.dataset.editorNode = "true";
    element.tabIndex = 0;
  });

  const select = (element) => {
    if (selected) selected.classList.remove("editor-selected");
    selected = element || null;
    if (selected) {
      selected.classList.add("editor-selected");
      setStatus(selected.dataset.editorObject ? "Image selected" : "Text selected");
    } else {
      setStatus("Ready");
    }
  };

  const placeAbsolutely = (element) => {
    const pageStyle = window.getComputedStyle(page);
    if (pageStyle.position === "static") page.style.position = "relative";
    const pageBox = page.getBoundingClientRect();
    const box = element.getBoundingClientRect();
    if (window.getComputedStyle(element).position !== "absolute") {
      element.style.position = "absolute";
      element.style.left = `${Math.max(0, box.left - pageBox.left)}px`;
      element.style.top = `${Math.max(0, box.top - pageBox.top)}px`;
      element.style.margin = "0";
      element.style.zIndex = "10";
    }
  };

  const nudge = (x, y) => {
    if (!selected) return setStatus("Select text or an image first");
    placeAbsolutely(selected);
    selected.style.left = `${(parseFloat(selected.style.left) || 0) + x}px`;
    selected.style.top = `${(parseFloat(selected.style.top) || 0) + y}px`;
    setStatus("Moved");
  };

  const scale = (factor) => {
    if (!selected) return setStatus("Select text or an image first");
    if (selected.dataset.editorObject === "true") {
      const box = selected.getBoundingClientRect();
      selected.style.width = `${Math.max(16, box.width * factor)}px`;
      selected.style.height = "auto";
    } else {
      const fontSize = parseFloat(window.getComputedStyle(selected).fontSize) || 16;
      selected.style.fontSize = `${Math.max(8, fontSize * factor)}px`;
    }
    setStatus("Resized");
  };

  const addText = () => {
    const element = document.createElement("div");
    element.textContent = "New text";
    element.dataset.editorText = "true";
    element.dataset.editorNode = "true";
    element.contentEditable = "true";
    element.spellcheck = false;
    element.style.cssText = "position:absolute;left:24px;top:24px;z-index:10;padding:2px 4px;color:#111827;font:600 18px/1.2 Arial,sans-serif;";
    if (window.getComputedStyle(page).position === "static") page.style.position = "relative";
    page.appendChild(element);
    select(element);
    element.focus();
    setStatus("New text added");
  };

  const download = () => {
    const clone = document.documentElement.cloneNode(true);
    clone.querySelectorAll(".editor-selected").forEach((element) => element.classList.remove("editor-selected"));
    const source = "<!DOCTYPE html>\\n" + clone.outerHTML;
    const url = URL.createObjectURL(new Blob([source], { type: "text/html;charset=utf-8" }));
    const link = document.createElement("a");
    link.href = url;
    link.download = "clean_pad_edited.html";
    link.click();
    URL.revokeObjectURL(url);
    setStatus("Edited HTML downloaded");
  };

  page.addEventListener("click", (event) => {
    const element = event.target.closest("[data-editor-node]");
    if (element) select(element);
  });

  page.addEventListener("pointerdown", (event) => {
    const element = event.target.closest("[data-editor-object]");
    if (!element) return;
    event.preventDefault();
    select(element);
    placeAbsolutely(element);
    drag = {
      pointerId: event.pointerId,
      element,
      x: event.clientX,
      y: event.clientY,
      left: parseFloat(element.style.left) || 0,
      top: parseFloat(element.style.top) || 0,
    };
    element.setPointerCapture(event.pointerId);
  });

  page.addEventListener("pointermove", (event) => {
    if (!drag || event.pointerId !== drag.pointerId) return;
    drag.element.style.left = `${drag.left + event.clientX - drag.x}px`;
    drag.element.style.top = `${drag.top + event.clientY - drag.y}px`;
  });

  page.addEventListener("pointerup", (event) => {
    if (drag && event.pointerId === drag.pointerId) {
      setStatus("Image moved");
      drag = null;
    }
  });

  toolbar.addEventListener("click", (event) => {
    const action = event.target.closest("button")?.dataset.editorAction;
    if (!action) return;
    if (action === "add-text") addText();
    if (action === "smaller") scale(0.9);
    if (action === "larger") scale(1.1);
    if (action === "left") nudge(-5, 0);
    if (action === "up") nudge(0, -5);
    if (action === "down") nudge(0, 5);
    if (action === "right") nudge(5, 0);
    if (action === "delete") {
      if (!selected) return setStatus("Select text or an image first");
      selected.remove();
      select(null);
      setStatus("Deleted");
    }
    if (action === "print") window.print();
    if (action === "download") download();
  });

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") select(null);
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "s") {
      event.preventDefault();
      download();
    }
  });
})();
</script>
"""


def build_editor_html(reconstructed_html: str) -> str:
    """Return a self-contained editable version of a reconstructed document."""
    if EDITOR_MARKER in reconstructed_html:
        return reconstructed_html
    if not re.search(r"</head\s*>", reconstructed_html, flags=re.IGNORECASE):
        raise BrowserEditorError("Generated HTML does not contain a closing </head> tag.")
    if not re.search(r"</body\s*>", reconstructed_html, flags=re.IGNORECASE):
        raise BrowserEditorError("Generated HTML does not contain a closing </body> tag.")

    with_style = re.sub(
        r"</head\s*>",
        lambda _match: EDITOR_STYLE + "</head>",
        reconstructed_html,
        count=1,
        flags=re.IGNORECASE,
    )
    return re.sub(
        r"</body\s*>",
        lambda _match: EDITOR_UI + "</body>",
        with_style,
        count=1,
        flags=re.IGNORECASE,
    )


def publish_editor_html(editor_html: str, directory: Path = EDITOR_DIRECTORY) -> str:
    """Write the editor document to the app static directory and return its URL.

    A content hash makes each reconstructed result deterministic and prevents a
    filename from exposing the uploaded prescription's original name.
    """
    if EDITOR_MARKER not in editor_html:
        raise BrowserEditorError("Only browser-editor HTML can be published.")

    digest = hashlib.sha256(editor_html.encode("utf-8")).hexdigest()[:20]
    filename = f"prescription-{digest}.html"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    path.write_text(editor_html, encoding="utf-8")
    return f"{EDITOR_URL_PREFIX}/{filename}"
