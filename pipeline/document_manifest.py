"""Validated intermediate representation for reconstructed documents."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
import re
from typing import Any


ALLOWED_KINDS = {"text", "image", "line", "shape"}


def _number(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _color(value: Any, default: str = "#000000") -> str:
    text = str(value or default).strip()
    if re.fullmatch(r"#[0-9a-fA-F]{3}(?:[0-9a-fA-F]{3})?", text):
        return text
    if re.fullmatch(r"rgba?\([^)]*\)", text):
        return text
    return default


@dataclass(frozen=True)
class NormalizedBox:
    x: float
    y: float
    width: float
    height: float

    @classmethod
    def from_value(cls, value: Any) -> "NormalizedBox":
        if isinstance(value, dict):
            values = [value.get(key) for key in ("x", "y", "width", "height")]
        elif isinstance(value, (list, tuple)) and len(value) == 4:
            values = list(value)
        else:
            values = [0, 0, 1, 0.05]
        x, y, width, height = (_number(item, 0.0) for item in values)
        x = min(1.0, max(0.0, x))
        y = min(1.0, max(0.0, y))
        width = min(1.0 - x, max(0.0005, width))
        height = min(1.0 - y, max(0.0005, height))
        return cls(x, y, width, height)

    def transformed(
        self, scale_x: float = 1.0, scale_y: float = 1.0, dx: float = 0.0, dy: float = 0.0
    ) -> "NormalizedBox":
        return NormalizedBox.from_value(
            [self.x * scale_x + dx, self.y * scale_y + dy, self.width * scale_x, self.height * scale_y]
        )


@dataclass(frozen=True)
class DocumentElement:
    id: str
    kind: str
    box: NormalizedBox
    text: str = ""
    role: str = ""
    font_family: str = "Nirmala UI"
    font_size: float = 0.015
    font_weight: int = 400
    line_height: float = 1.15
    color: str = "#000000"
    background: str = "transparent"
    opacity: float = 1.0
    align: str = "left"
    border_color: str = "#000000"
    border_width: float = 0.0
    rotation: float = 0.0
    z_index: int = 1

    @classmethod
    def from_dict(cls, data: dict[str, Any], index: int) -> "DocumentElement":
        kind = str(data.get("kind", "text")).lower()
        if kind not in ALLOWED_KINDS:
            kind = "text"
        align = str(data.get("align", "left")).lower()
        if align not in {"left", "center", "right", "justify"}:
            align = "left"
        weight = round(_number(data.get("font_weight"), 400))
        weight = min(900, max(100, weight))
        return cls(
            id=re.sub(r"[^a-zA-Z0-9_-]", "-", str(data.get("id") or f"element-{index}")),
            kind=kind,
            box=NormalizedBox.from_value(data.get("bbox") or data.get("box")),
            text=str(data.get("text") or ""),
            role=str(data.get("role") or ""),
            font_family=str(data.get("font_family") or "Nirmala UI"),
            font_size=min(0.2, max(0.003, _number(data.get("font_size"), 0.015))),
            font_weight=weight,
            line_height=min(3.0, max(0.7, _number(data.get("line_height"), 1.15))),
            color=_color(data.get("color")),
            background=_color(data.get("background"), "transparent")
            if data.get("background") not in (None, "transparent")
            else "transparent",
            opacity=min(1.0, max(0.02, _number(data.get("opacity"), 1.0))),
            align=align,
            border_color=_color(data.get("border_color")),
            border_width=min(0.02, max(0.0, _number(data.get("border_width"), 0.0))),
            rotation=min(180.0, max(-180.0, _number(data.get("rotation"), 0.0))),
            z_index=round(_number(data.get("z_index"), index + 1)),
        )


@dataclass(frozen=True)
class DocumentManifest:
    background: str = "#ffffff"
    elements: tuple[DocumentElement, ...] = field(default_factory=tuple)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DocumentManifest":
        raw_elements = data.get("elements")
        if not isinstance(raw_elements, list) or not raw_elements:
            raise ValueError("Document analysis did not contain any elements.")
        elements = tuple(
            DocumentElement.from_dict(item, index)
            for index, item in enumerate(raw_elements)
            if isinstance(item, dict)
        )
        if not elements:
            raise ValueError("Document analysis did not contain valid elements.")
        return cls(background=_color(data.get("background"), "#ffffff"), elements=elements)

    @classmethod
    def from_json(cls, raw: str) -> "DocumentManifest":
        text = raw.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("Document analysis did not return a JSON object.")
        return cls.from_dict(json.loads(text[start : end + 1]))

    def transform_all(
        self, scale_x: float = 1.0, scale_y: float = 1.0, dx: float = 0.0, dy: float = 0.0
    ) -> "DocumentManifest":
        return replace(
            self,
            elements=tuple(
                replace(element, box=element.box.transformed(scale_x, scale_y, dx, dy))
                for element in self.elements
            ),
        )

