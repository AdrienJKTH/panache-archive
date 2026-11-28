"""What the daily step of Panache's 1 km maps of the present has to do (Panache's P7).

The maps are `.days` files of the private dataset's folder `downscaled-live-v1`, named after their
days: `downscaled-<pollutant>-present-<first>-to-<last>.days`. The month under way gets a file a
pass (a day, or the days a missed pass left); once the month is over, its files are joined into
one. A published file is never rewritten, since the quota counts every revision. This module only
reads the folder's file names and says what is missing; the workflow builds, joins and deletes.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from itertools import pairwise
from typing import Final

ASSET: Final = re.compile(
    r"downscaled-(?P<pollutant>\w+)-present-(?P<first>\d{4}-\d\d-\d\d)"
    r"-to-(?P<last>\d{4}-\d\d-\d\d)\.days"
)


@dataclass(frozen=True, slots=True, order=True)
class Span:
    """A pollutant's days, and the file that holds or will hold them."""

    pollutant: str
    first: date
    last: date

    @property
    def name(self) -> str:
        return f"downscaled-{self.pollutant}-present-{self.first}-to-{self.last}.days"


@dataclass(frozen=True, slots=True)
class Join:
    """A month's files to make one, in the order of their days."""

    whole: Span
    parts: tuple[Span, ...]


def present_files(names: Sequence[str]) -> list[Span]:
    """The present's files among `names`, by pollutant and days."""
    return sorted(
        Span(
            match["pollutant"],
            date.fromisoformat(match["first"]),
            date.fromisoformat(match["last"]),
        )
        for name in names
        if (match := ASSET.fullmatch(name.strip()))
    )


def month_end(day: date) -> date:
    return (day.replace(day=1) + timedelta(days=32)).replace(day=1) - timedelta(days=1)


def missing(names: Sequence[str], today: date) -> list[Span]:
    """The days after each pollutant's last file up to yesterday, a file a month at most. A
    pollutant without any file is left to the catch-up."""
    files = present_files(names)
    found = []
    yesterday = today - timedelta(days=1)
    for pollutant in sorted({f.pollutant for f in files}):
        first = max(f.last for f in files if f.pollutant == pollutant) + timedelta(days=1)
        while first <= yesterday:
            last = min(month_end(first), yesterday)
            found.append(Span(pollutant, first, last))
            first = last + timedelta(days=1)
    return found


def joins(names: Sequence[str], today: date) -> list[Join]:
    """The months over (their last day before `today`) held in several files that follow one
    another from the month's first day to its last."""
    months: dict[tuple[str, date], list[Span]] = {}
    for file in present_files(names):
        months.setdefault((file.pollutant, file.first.replace(day=1)), []).append(file)
    found = []
    for (pollutant, first), parts in sorted(months.items()):
        last = month_end(first)
        if last >= today or len(parts) < 2:
            continue
        follow = all(b.first == a.last + timedelta(days=1) for a, b in pairwise(parts))
        if follow and parts[0].first == first and parts[-1].last == last:
            found.append(Join(Span(pollutant, first, last), tuple(parts)))
    return found
