"""Copies a public release into the private dataset (`panache_archive.private`), a batch at a time.

What Panache computes was published in GitHub releases before the private dataset took it: the
maps of 14 and 58 GB, among others. A runner's disk holds one batch of `mirror.BATCH_BYTES` at a
time: each batch of assets is downloaded, checked against the release's SHA-256, uploaded in one
commit, and deleted. An asset already in the folder with the same size is left as it is, so that a
run cut short starts again where it stopped. Nothing is ever deleted from either side.
"""

import logging
from collections.abc import Sequence
from pathlib import Path

from panache_archive import mirror
from panache_archive.mirror import Asset, Download
from panache_archive.private import PrivateStore

logger = logging.getLogger(__name__)


def missing(release: Sequence[Asset], present: dict[str, int]) -> list[Asset]:
    """The release's assets the folder lacks, or holds with another size; the transient parts of
    a year being built stay out, as they do from the mirror."""
    return [
        asset
        for asset in sorted(release, key=lambda a: a.name)
        if not mirror.TRANSIENT.search(asset.name) and present.get(asset.name) != asset.size
    ]


def move(
    store: PrivateStore,
    release: Sequence[Asset],
    folder: str,
    workdir: Path,
    download: Download = mirror.download_asset,
    batch_bytes: int = mirror.BATCH_BYTES,
) -> list[Asset]:
    """Copies what `folder` lacks of `release` into it; returns the assets copied."""
    todo = missing(release, store.files(folder))
    workdir.mkdir(parents=True, exist_ok=True)
    for group in mirror.batches(todo, batch_bytes):
        paths = []
        for asset in group:
            target = workdir / asset.name
            download(asset, target)
            if asset.sha256 is not None and mirror.sha256(target) != asset.sha256:
                raise ValueError(f"{asset.name} was not downloaded whole")
            paths.append(target)
        store.upload(folder, paths)
        for path in paths:
            path.unlink()
        logger.info("moved %d assets of %s", len(group), folder)
    return todo
