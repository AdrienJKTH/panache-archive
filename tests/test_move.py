import hashlib
from pathlib import Path

import pytest

from panache_archive import cli, move, private
from panache_archive.mirror import Asset
from test_private import FakeApi

CONTENTS = {
    "downscaled-pm10-2020.days": b"pm10 of 2020",
    "downscaled-pm10-2021.days": b"pm10 of 2021",
    "downscaled-pm10-2021-part1.days": b"a part",
    "downscaling-pm10.lgb.txt": b"model",
}


def _asset(name: str, content: bytes | None = None) -> Asset:
    content = CONTENTS[name] if content is None else content
    return Asset(name, len(content), hashlib.sha256(content).hexdigest(), f"https://gh/{name}")


def _download(asset: Asset, target: Path) -> None:
    target.write_bytes(CONTENTS[asset.name])


def test_what_the_folder_has_is_left_as_it_is_and_the_parts_stay_out() -> None:
    release = [_asset(name) for name in CONTENTS]
    present = {
        "downscaled-pm10-2020.days": len(CONTENTS["downscaled-pm10-2020.days"]),
        "downscaled-pm10-2021.days": 3,
    }

    assert [a.name for a in move.missing(release, present)] == [
        "downscaled-pm10-2021.days",  # another size
        "downscaling-pm10.lgb.txt",
    ]


def test_the_missing_assets_are_copied_in_batches_and_the_disk_is_freed(tmp_path: Path) -> None:
    api = FakeApi({"maps-v1/downscaled-pm10-2020.days": CONTENTS["downscaled-pm10-2020.days"]})
    release = [_asset(name) for name in CONTENTS]
    sizes: list[list[str]] = []
    store = private.PrivateStore("token", api=api)
    upload = store.upload

    def spy(folder: str, paths: list[Path]) -> None:
        sizes.append([p.name for p in paths])
        assert all(p.exists() for p in paths)
        upload(folder, paths)

    store.upload = spy  # type: ignore[method-assign,assignment]

    copied = move.move(store, release, "maps-v1", tmp_path / "work", _download, batch_bytes=15)

    assert [a.name for a in copied] == ["downscaled-pm10-2021.days", "downscaling-pm10.lgb.txt"]
    assert sizes == [["downscaled-pm10-2021.days"], ["downscaling-pm10.lgb.txt"]]
    assert list((tmp_path / "work").iterdir()) == []
    assert [len(ops) for _, ops, _ in api.commits] == [1, 1]


def test_a_download_that_is_cut_short_is_never_uploaded(tmp_path: Path) -> None:
    api = FakeApi()
    release = [_asset("downscaling-pm10.lgb.txt")]

    def cut(asset: Asset, target: Path) -> None:
        target.write_bytes(CONTENTS[asset.name][:-1])

    with pytest.raises(ValueError, match="whole"):
        move.move(private.PrivateStore("token", api=api), release, "maps-v1", tmp_path, cut)

    assert api.commits == []


def test_the_import_command_copies_a_release_json(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    listing = tmp_path / "release.json"
    listing.write_text(
        '{"assets": [{"name": "downscaling-pm10.lgb.txt", "size": 5, "browser_download_url": "u"}]}'
    )
    asked: list[tuple[str, list[str]]] = []

    def stub(store: object, release: list[Asset], folder: str, workdir: Path) -> list[Asset]:
        assert isinstance(store, private.PrivateStore)
        assert workdir.is_dir()
        asked.append((folder, [a.name for a in release]))
        return release

    monkeypatch.setenv("HF_TOKEN", "token")
    monkeypatch.setattr(private, "HfApi", lambda **_: FakeApi())
    monkeypatch.setattr(move, "move", stub)

    code = cli.main(["private", "import", "--release", str(listing), "--folder", "maps-v1"])

    assert code == 0
    assert asked == [("maps-v1", ["downscaling-pm10.lgb.txt"])]
    assert "1 assets in the release: 1 copied into maps-v1" in capsys.readouterr().out
