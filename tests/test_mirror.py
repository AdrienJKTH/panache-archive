import hashlib
import io
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

import pytest
from huggingface_hub import CommitOperationAdd, CommitOperationDelete
from huggingface_hub.hf_api import RepoFile

from panache_archive import cli, mirror
from panache_archive.mirror import Asset, git_blob_id

CARD = b"---\nlicense: cc-by-4.0\n---\n# The card\n"


def _asset(name: str, content: bytes) -> Asset:
    return Asset(name, len(content), hashlib.sha256(content).hexdigest(), f"https://gh/{name}")


CONTENTS = {
    "cams-pm10-2021-validated.cells": b"pm10 of 2021",
    "cams-pm10-2025-validated.cells": b"pm10 of 2025, validated",
    "cams-ozone-2021-validated.cells": b"ozone of 2021",
}
RELEASE = [_asset(name, content) for name, content in CONTENTS.items()]


class FakeHub:
    def __init__(self, files: Sequence[Asset] = (), card: str | None = None) -> None:
        self.mirrored = list(files)
        self.card = card
        self.commits: list[tuple[dict[str, bytes], list[str], str]] = []
        self.folders: list[str] = []

    def files(self, folder: str) -> tuple[list[Asset], str | None]:
        self.folders.append(folder)
        return self.mirrored, self.card

    def commit(
        self, uploads: Mapping[str, Path | bytes], deletes: Sequence[str], message: str
    ) -> None:
        read = {p: c if isinstance(c, bytes) else c.read_bytes() for p, c in uploads.items()}
        self.commits.append((read, list(deletes), message))


def _download(asset: Asset, target: Path) -> None:
    target.write_bytes(CONTENTS[asset.name])


def test_the_release_lists_its_cells_with_their_digest() -> None:
    listing = [
        {"name": "a.cells", "size": 3, "digest": "sha256:abc", "browser_download_url": "u"},
        {"name": "b.cells", "size": 4, "digest": None},
        {"name": "notes.md", "size": 1},
    ]

    assert mirror.release_assets(listing) == [Asset("a.cells", 3, "abc", "u"), Asset("b.cells", 4)]


def test_the_plan_uploads_what_differs_and_deletes_what_the_release_dropped() -> None:
    mirrored = [
        RELEASE[0],
        Asset(RELEASE[1].name, RELEASE[1].size + 1, RELEASE[1].sha256),  # another size
        Asset("cams-pm10-2025-interim.cells", 5, "old"),
    ]

    todo = mirror.plan(RELEASE, mirrored, CARD, git_blob_id(CARD))

    assert [a.name for a in todo.upload] == [
        "cams-ozone-2021-validated.cells",
        "cams-pm10-2025-validated.cells",
    ]
    assert todo.delete == ("cams-pm10-2025-interim.cells",)
    assert not todo.card


@pytest.mark.parametrize(
    ("mirrored_sha", "uploaded"), [("other", True), (None, False), ("same", False)]
)
def test_a_digest_decides_when_both_sides_have_one(
    mirrored_sha: str | None, uploaded: bool
) -> None:
    asset = Asset("a.cells", 3, "same")

    todo = mirror.plan([asset], [Asset("a.cells", 3, mirrored_sha)], CARD, None)

    assert bool(todo.upload) is uploaded
    assert todo.card  # no card on the mirror yet


def test_a_mirror_in_line_has_nothing_to_do() -> None:
    assert mirror.plan(RELEASE, RELEASE, CARD, git_blob_id(CARD)).empty


def test_batches_stay_under_their_limit() -> None:
    sizes = [Asset(str(i), size) for i, size in enumerate([3, 3, 5, 1, 9])]

    groups = mirror.batches(sizes, limit=6)

    assert [[a.size for a in group] for group in groups] == [[3, 3], [5, 1], [9]]
    assert mirror.batches([]) == []


def test_sync_writes_the_card_deletes_then_uploads_in_batches(tmp_path: Path) -> None:
    hub = FakeHub([Asset("cams-pm10-2024-interim.cells", 5)])

    done = mirror.sync(hub, RELEASE, CARD, tmp_path / "work", _download)

    assert len(done.upload) == 3
    first, upload = hub.commits
    assert first[0] == {"README.md": CARD}
    assert first[1] == ["cams-europe-v1/cams-pm10-2024-interim.cells"]
    assert upload[0] == {
        f"cams-europe-v1/{name}": content for name, content in sorted(CONTENTS.items())
    }
    assert upload[1] == []
    assert list((tmp_path / "work").iterdir()) == []  # each batch is deleted once committed


def test_sync_in_line_commits_nothing(tmp_path: Path) -> None:
    hub = FakeHub(RELEASE, git_blob_id(CARD))

    assert mirror.sync(hub, RELEASE, CARD, tmp_path, _download).empty
    assert hub.commits == []


def test_a_download_cut_short_is_never_committed(tmp_path: Path) -> None:
    def cut(asset: Asset, target: Path) -> None:
        target.write_bytes(CONTENTS[asset.name][:-1])

    hub = FakeHub([], git_blob_id(CARD))

    with pytest.raises(ValueError, match="not downloaded whole"):
        mirror.sync(hub, RELEASE, CARD, tmp_path, cut)
    assert hub.commits == []


def test_the_card_is_known_by_its_git_blob_id() -> None:
    assert git_blob_id(b"hello\n") == "ce013625030ba8dba906f756967f9e9ca394464a"


def test_an_asset_is_streamed_from_its_url(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    asked: list[str] = []

    def urlopen(url: str, timeout: float) -> io.BytesIO:
        asked.append(url)
        assert timeout > 0
        return io.BytesIO(b"the bytes")

    monkeypatch.setattr("urllib.request.urlopen", urlopen)

    mirror.download_asset(Asset("a.cells", 9, url="https://gh/a.cells"), tmp_path / "a.cells")

    assert (tmp_path / "a.cells").read_bytes() == b"the bytes"
    assert asked == ["https://gh/a.cells"]


def _file(**entry: object) -> RepoFile:
    """A file as Hugging Face's API lists it."""
    return RepoFile(**entry)  # type: ignore[no-untyped-call]


class FakeHfApi:
    def __init__(self, token: str) -> None:
        self.token = token
        self.commits: list[tuple[str, list[object], str, str]] = []

    def list_repo_tree(self, repo_id: str, *, repo_type: str, recursive: bool) -> list[object]:
        assert (repo_id, repo_type, recursive) == (mirror.REPO_ID, "dataset", True)
        return [
            _file(path="README.md", size=10, oid="cardblob"),
            _file(path=".gitattributes", size=2, oid="x"),
            _file(
                path="cams-europe-v1/a.cells",
                size=3,
                oid="y",
                lfs={"size": 3, "oid": "abc", "pointerSize": 130},
            ),
        ]

    def create_commit(
        self, repo_id: str, operations: list[object], *, commit_message: str, repo_type: str
    ) -> None:
        self.commits.append((repo_id, operations, commit_message, repo_type))


def test_the_hub_lists_the_folder_and_commits_through_huggingface_hub(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(mirror, "HfApi", FakeHfApi)
    hub = mirror.HubApi("token")
    path = tmp_path / "a.cells"
    path.write_bytes(b"abc")

    files, card = hub.files("cams-europe-v1")
    hub.commit({"README.md": CARD, "cams-europe-v1/a.cells": path}, ["cams-europe-v1/b.cells"], "m")

    assert files == [Asset("a.cells", 3, "abc")]
    assert card == "cardblob"
    api = cast("FakeHfApi", hub._api)
    ((repo, operations, message, kind),) = api.commits
    assert (repo, message, kind) == (mirror.REPO_ID, "m", "dataset")
    assert [type(o) for o in operations] == [
        CommitOperationAdd,
        CommitOperationAdd,
        CommitOperationDelete,
    ]


def test_mirror_needs_a_token(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)

    code = cli.main(["mirror", "--release", str(tmp_path / "r.json"), "--card", "c.md"])

    assert code == 2


def test_mirror_syncs_the_release_with_the_hub(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    release = tmp_path / "release.json"
    release.write_text(
        json.dumps({"assets": [{"name": "a.cells", "size": 3, "digest": None}]}), encoding="utf-8"
    )
    card = tmp_path / "card.md"
    card.write_bytes(CARD)
    synced: list[tuple[list[Asset], bytes]] = []

    def sync(
        _hub: object, assets: list[Asset], content: bytes, *_: object, folder: str
    ) -> mirror.Plan:
        assert folder == "cams-europe-v1"
        synced.append((assets, content))
        return mirror.Plan(upload=tuple(assets), delete=(), card=True)

    monkeypatch.setenv("HF_TOKEN", "token")
    monkeypatch.setattr(mirror, "HfApi", FakeHfApi)
    monkeypatch.setattr(mirror, "sync", sync)

    code = cli.main(["mirror", "--release", str(release), "--card", str(card)])

    assert code == 0
    assert synced == [([Asset("a.cells", 3)], CARD)]
    assert (
        "1 assets in the release: 1 uploaded, 0 deleted, the card written"
        in capsys.readouterr().out
    )


def test_each_release_has_its_folder(tmp_path: Path) -> None:
    hub = FakeHub([], git_blob_id(CARD))

    mirror.sync(hub, RELEASE[:1], CARD, tmp_path, _download, folder="era5-europe-v1")

    assert hub.folders == ["era5-europe-v1"]
    ((uploads, _, message),) = hub.commits
    assert list(uploads) == ["era5-europe-v1/cams-pm10-2021-validated.cells"]
    assert message.startswith("Mirror era5-europe-v1:")
