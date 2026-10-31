"""The Hugging Face mirror of the whole domain's release (Panache's D66).

Panache reads the whole domain's assets from GitHub's release `cams-europe-v1`, and from their
mirror on Hugging Face when GitHub fails: the public dataset `adrien4916/air_qual_indicators`,
folder `cams-europe-v1`. The `Mirror` workflow keeps the folder equal to the release:
- an asset missing from the mirror, or of another size or SHA-256, is uploaded again;
- a file the release no longer holds (an interim year replaced by its validated twin) is deleted;
- the dataset card (`README.md`) follows `docs/hugging-face-card.md`.
Assets go in batches of at most `BATCH_BYTES`, one commit each, so that a runner's disk holds one
batch at a time.
"""

import hashlib
import json
import logging
import shutil
import urllib.request
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Protocol

from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi
from huggingface_hub.hf_api import RepoFile

from panache_archive import web

REPO_ID: Final = "adrien4916/air_qual_indicators"
FOLDER: Final = "cams-europe-v1"
"""The first release mirrored; each release has a folder of its name (`era5-europe-v1`)."""
CARD: Final = "README.md"
BATCH_BYTES: Final = 4_000_000_000
"""What a batch downloads at most: ~10 assets, well within a runner's disk."""

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Asset:
    """A file on either side: its name, size, and SHA-256 when known."""

    name: str
    size: int
    sha256: str | None = None
    url: str = ""
    """Where GitHub serves it (release assets only)."""


@dataclass(frozen=True, slots=True)
class Plan:
    upload: tuple[Asset, ...]
    delete: tuple[str, ...]
    card: bool
    """Whether the dataset card must be written."""

    @property
    def empty(self) -> bool:
        return not (self.upload or self.delete or self.card)


class Hub(Protocol):
    """What the mirror needs of Hugging Face (`HubApi`, or a fake)."""

    def files(self, folder: str) -> tuple[list[Asset], str | None]:
        """The folder's files, and the card's git blob id (`None` without a card)."""
        ...

    def commit(
        self, uploads: Mapping[str, Path | bytes], deletes: Sequence[str], message: str
    ) -> None:
        """One commit adding (by path in the repository) and deleting files."""
        ...


Download = Callable[[Asset, Path], None]
"""Fetches a release asset into a file."""


def release_assets(listing: Iterable[Mapping[str, Any]]) -> list[Asset]:
    """The release's `.cells` assets, from GitHub's JSON (`name`, `size`, `digest`, and
    `browser_download_url`)."""
    assets = []
    for entry in listing:
        name = str(entry["name"])
        if not name.endswith(".cells"):
            continue
        digest = entry.get("digest") or ""
        sha256 = digest.removeprefix("sha256:") if digest.startswith("sha256:") else None
        assets.append(
            Asset(name, int(entry["size"]), sha256, str(entry.get("browser_download_url", "")))
        )
    return assets


def plan(
    release: Sequence[Asset], mirrored: Sequence[Asset], card: bytes, card_blob: str | None
) -> Plan:
    """What to upload and delete so that the mirror holds exactly the release, and its card."""
    there = {a.name: a for a in mirrored}
    upload = tuple(
        asset
        for asset in sorted(release, key=lambda a: a.name)
        if asset.name not in there or not _same(asset, there[asset.name])
    )
    names = {a.name for a in release}
    delete = tuple(sorted(name for name in there if name not in names))
    return Plan(upload=upload, delete=delete, card=git_blob_id(card) != card_blob)


def batches(assets: Sequence[Asset], limit: int = BATCH_BYTES) -> list[list[Asset]]:
    """Consecutive groups of assets, each of at most `limit` bytes (or one larger asset)."""
    groups: list[list[Asset]] = []
    size = 0
    for asset in assets:
        if not groups or size + asset.size > limit:
            groups.append([])
            size = 0
        groups[-1].append(asset)
        size += asset.size
    return groups


def sync(
    hub: Hub,
    release: Sequence[Asset],
    card: bytes,
    workdir: Path,
    download: Download,
    folder: str = FOLDER,
) -> Plan:
    """Bring the mirror's `folder` in line with `release`, and its card; returns what was done."""
    mirrored, card_blob = hub.files(folder)
    todo = plan(release, mirrored, card, card_blob)
    if todo.card or todo.delete:
        hub.commit(
            {CARD: card} if todo.card else {},
            [f"{folder}/{name}" for name in todo.delete],
            f"Mirror {folder}: the card, and {len(todo.delete)} assets the release dropped",
        )
    workdir.mkdir(parents=True, exist_ok=True)
    for group in batches(todo.upload):
        paths: dict[str, Path | bytes] = {}
        for asset in group:
            target = workdir / asset.name
            download(asset, target)
            if asset.sha256 is not None and _sha256(target) != asset.sha256:
                raise ValueError(f"{asset.name} was not downloaded whole")
            paths[f"{folder}/{asset.name}"] = target
        hub.commit(paths, [], f"Mirror {folder}: {', '.join(a.name for a in group)}")
        for path in paths.values():
            if isinstance(path, Path):
                path.unlink()
        logger.info("mirrored %d assets", len(group))
    return todo


def git_blob_id(content: bytes) -> str:
    """The git blob id of a file, as Hugging Face lists files that are not in LFS."""
    return hashlib.sha1(b"blob %d\0" % len(content) + content).hexdigest()


def download_asset(asset: Asset, target: Path) -> None:
    """Stream a release asset to `target` (GitHub redirects to its storage)."""
    with (
        urllib.request.urlopen(asset.url, timeout=web.TIMEOUT_SECONDS) as answer,
        target.open("wb") as out,
    ):
        shutil.copyfileobj(answer, out, length=1 << 20)


class HubApi:
    """The mirror's dataset through `huggingface_hub`."""

    def __init__(self, token: str, repo_id: str = REPO_ID) -> None:
        self._api = HfApi(token=token)
        self._repo_id = repo_id

    def files(self, folder: str) -> tuple[list[Asset], str | None]:
        entries = list(self._api.list_repo_tree(self._repo_id, repo_type="dataset", recursive=True))
        files = [e for e in entries if isinstance(e, RepoFile)]
        mirrored = [
            Asset(e.path.removeprefix(f"{folder}/"), e.size, e.lfs.sha256 if e.lfs else None)
            for e in files
            if e.path.startswith(f"{folder}/")
        ]
        card = next((e.blob_id for e in files if e.path == CARD), None)
        return mirrored, card

    def commit(
        self, uploads: Mapping[str, Path | bytes], deletes: Sequence[str], message: str
    ) -> None:
        operations: list[CommitOperationAdd | CommitOperationDelete] = [
            CommitOperationAdd(
                path_in_repo=path,
                path_or_fileobj=content if isinstance(content, bytes) else str(content),
            )
            for path, content in uploads.items()
        ]
        operations += [CommitOperationDelete(path_in_repo=path) for path in deletes]
        self._api.create_commit(
            self._repo_id, operations, commit_message=message, repo_type="dataset"
        )


def _same(asset: Asset, mirrored: Asset) -> bool:
    if asset.size != mirrored.size:
        return False
    return asset.sha256 is None or mirrored.sha256 is None or asset.sha256 == mirrored.sha256


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_release(path: Path) -> list[Asset]:
    """The release's assets from the JSON the workflow saved (`gh api .../releases/tags/...`)."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return release_assets(data.get("assets", []) if isinstance(data, dict) else data)
