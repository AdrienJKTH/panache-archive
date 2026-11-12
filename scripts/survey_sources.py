"""Survey of the two open source-apportionment targets (Panache's feature 4, public version):
what their Zenodo files really hold, written as a plain-text report.

Usage: python -I scripts/survey_sources.py <download dir> <report path>

Each record's files are listed from Zenodo's API and downloaded; each archive's members are
listed, and the first lines of every text member (or the first rows of every sheet) are printed.
Nothing is parsed: the report is read to write the intake that does.
"""

import io
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

RECORDS = {
    "chen-2022": "6672710",
    "ri-urbans-ebc": "7982201",
}
LINES = 12
TEXT = (".csv", ".txt", ".dat", ".tsv", ".md", ".json", ".nas", ".xml")


def main(download: Path, report: Path) -> None:
    out: list[str] = []
    for name, record in RECORDS.items():
        out.append(f"===== {name} (zenodo {record})")
        meta = json.loads(_read(f"https://zenodo.org/api/records/{record}"))
        out.append(f"title: {meta['metadata']['title']}")
        out.append(f"version: {meta['metadata'].get('version')}  doi: {meta.get('doi')}")
        out.append(f"license: {meta['metadata'].get('license')}")
        for entry in meta["files"]:
            url = entry["links"]["self"]
            key = entry["key"]
            out.append(f"--- file {key} {entry['size']} bytes {entry['checksum']}")
            path = download / name / key
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(_read(url))
            _describe(path.name, path.read_bytes(), out, depth=0)
    report.write_text("\n".join(out), encoding="utf-8")


def _describe(name: str, content: bytes, out: list[str], depth: int) -> None:
    pad = "  " * depth
    lower = name.lower()
    if lower.endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            members = [m for m in archive.infolist() if not m.is_dir()]
            out.append(f"{pad}zip {name}: {len(members)} members")
            for member in members:
                out.append(f"{pad}  {member.filename} {member.file_size}")
            for member in members:
                if "__MACOSX" in member.filename:
                    continue
                out.append(f"{pad}>>> {member.filename}")
                try:
                    _describe(member.filename, archive.read(member), out, depth + 1)
                except Exception as error:
                    out.append(f"{pad}  unreadable: {error!r}")
    elif lower.endswith(TEXT):
        text = content.decode("utf-8", errors="replace")
        lines = text.splitlines()
        out.append(f"{pad}{len(lines)} lines")
        out.extend(f"{pad}| {line[:400]}" for line in lines[:LINES])
        if len(lines) > LINES:
            out.append(f"{pad}| ... last: {lines[-1][:400]}")
    elif lower.endswith((".xlsx", ".xlsm")):
        import openpyxl

        book = openpyxl.load_workbook(io.BytesIO(content), read_only=True, data_only=True)
        for sheet in book.worksheets:
            out.append(f"{pad}sheet {sheet.title}: {sheet.max_row} rows x {sheet.max_column} cols")
            for row in sheet.iter_rows(max_row=LINES, values_only=True):
                out.append(f"{pad}| " + " ; ".join("" if v is None else str(v) for v in row)[:600])
    else:
        out.append(f"{pad}(binary, {len(content)} bytes, starts {content[:16]!r})")


def _read(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "panache-archive survey"})
    with urllib.request.urlopen(request, timeout=600) as response:
        return bytes(response.read())


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
