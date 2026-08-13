"""Preserve the source document's page geometry through reconstruction."""

from dataclasses import dataclass


PRINT_LONG_EDGE_MM = 297.0


@dataclass(frozen=True)
class PageGeometry:
    """Physical output dimensions for one reconstructed prescription page."""

    width_mm: float
    height_mm: float

    def __post_init__(self) -> None:
        if self.width_mm <= 0 or self.height_mm <= 0:
            raise ValueError("Page dimensions must be positive.")

    @property
    def orientation(self) -> str:
        if self.width_mm > self.height_mm:
            return "landscape"
        if self.height_mm > self.width_mm:
            return "portrait"
        return "square"

    @property
    def css_size(self) -> str:
        """CSS page-size value with stable precision."""
        return f"{self.width_mm:.3f}mm {self.height_mm:.3f}mm"

    @classmethod
    def from_pixels(
        cls, width_px: int, height_px: int, long_edge_mm: float = PRINT_LONG_EDGE_MM
    ) -> "PageGeometry":
        """Create a print page with the source image's displayed aspect ratio.

        Camera images rarely include trustworthy physical DPI metadata. We use
        an A4-length long edge for practical printing, while retaining the
        exact source ratio and orientation instead of forcing every upload to
        portrait A4.
        """
        if width_px <= 0 or height_px <= 0:
            raise ValueError("Image dimensions must be positive.")
        if long_edge_mm <= 0:
            raise ValueError("The print long edge must be positive.")

        longest_px = max(width_px, height_px)
        scale = long_edge_mm / longest_px
        return cls(width_px * scale, height_px * scale)
