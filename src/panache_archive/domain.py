"""The archive's two domains: where a release's assets lie, and how they are written.

France's assets are Parquet tables, one row per cell and day, for reading whole (Panache's D44).
The whole CAMS Europe grid is too large to read whole from afar, so its assets are cell-major
frames that give one cell in two small requests (`cells`, Panache's D66).
"""

from enum import StrEnum

from panache_archive.grid import EUROPE, FRANCE, Box


class Domain(StrEnum):
    FRANCE = "france"
    EUROPE = "europe"

    @property
    def box(self) -> Box:
        return FRANCE if self is Domain.FRANCE else EUROPE

    @property
    def suffix(self) -> str:
        """The extension of the domain's assets."""
        return ".parquet" if self is Domain.FRANCE else ".cells"
