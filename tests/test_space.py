from pathlib import Path

import pytest

from panache_archive import cli, space


def package(root: Path, files: dict[str, str]) -> Path:
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return root


@pytest.fixture
def source(tmp_path: Path) -> Path:
    return package(
        tmp_path / "src",
        {
            "panache/__init__.py": "",
            "panache/compute_app.py": (
                "import numpy\n"
                "import panache.config\n"
                "from panache.downscaling import fine\n"
                "from panache.sources.archive import ArchiveFiles\n"
            ),
            "panache/fetch_archive.py": "from panache.config import Settings\n",
            "panache/config.py": "",
            "panache/downscaling/__init__.py": "",
            "panache/downscaling/fine.py": "",
            "panache/sources/__init__.py": "",
            "panache/sources/archive.py": "from panache.sources import base\n",
            "panache/sources/base.py": "",
            "panache/verdict.py": "SECRET = 1\n",
            "panache/api/__init__.py": "",
            "panache/api/stories.py": "from panache.config import Settings\n",
        },
    )


def test_only_what_the_computation_imports_is_kept(source: Path) -> None:
    found = {path.as_posix() for path in space.imported_files(source)}

    assert found == {
        "panache/__init__.py",
        "panache/compute_app.py",
        "panache/config.py",
        "panache/downscaling/__init__.py",
        "panache/downscaling/fine.py",
        "panache/fetch_archive.py",
        "panache/sources/__init__.py",
        "panache/sources/archive.py",
        "panache/sources/base.py",
    }


def test_a_root_that_is_not_there_is_refused(source: Path) -> None:
    with pytest.raises(ValueError, match="nowhere"):
        space.imported_files(source, ["panache.nowhere"])


def test_the_bundle_holds_the_card_the_pins_and_the_code(source: Path, tmp_path: Path) -> None:
    card = package(tmp_path / "space", {"Dockerfile": "FROM python\n", "README.md": "card\n"})
    pins = tmp_path / "pins.txt"
    pins.write_text("numpy==2\n", encoding="utf-8")

    files = space.bundle(source, card, pins, tmp_path / "out")

    out = tmp_path / "out"
    assert (out / "Dockerfile").exists()
    assert (out / "requirements.txt").read_text(encoding="utf-8") == "numpy==2\n"
    assert (out / "panache" / "compute_app.py").exists()
    assert not (out / "panache" / "verdict.py").exists()
    assert not (out / "panache" / "api").exists()
    assert len(files) == 9


class FakeHub:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    def create_repo(self, repo_id: str, **kwargs: object) -> None:
        self.calls.append(("create", {"repo_id": repo_id, **kwargs}))

    def upload_folder(self, **kwargs: object) -> None:
        self.calls.append(("upload", kwargs))


def test_deploying_creates_the_space_then_replaces_its_files(tmp_path: Path) -> None:
    hub = FakeHub()

    space.deploy(hub, tmp_path)

    created, uploaded = hub.calls
    assert created[1]["repo_id"] == space.REPOSITORY
    assert created[1]["space_sdk"] == "docker"
    assert uploaded[1]["folder_path"] == tmp_path
    assert uploaded[1]["delete_patterns"] == "*"


def test_the_cli_needs_a_token(
    source: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    args = ["space", "--source", str(source), "--space", str(tmp_path), "--requirements", "r.txt"]

    assert cli.main([*args, "--out", str(tmp_path / "out")]) == 2


def test_the_cli_bundles_and_deploys(
    source: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("HF_TOKEN", "t")
    hub = FakeHub()
    monkeypatch.setattr("panache_archive.cli.HfApi", lambda token: hub)
    card = package(tmp_path / "space", {"Dockerfile": "FROM python\n"})
    pins = tmp_path / "pins.txt"
    pins.write_text("numpy==2\n", encoding="utf-8")
    args = ["space", "--source", str(source), "--space", str(card), "--requirements", str(pins)]

    assert cli.main([*args, "--out", str(tmp_path / "out")]) == 0

    assert [name for name, _ in hub.calls] == ["create", "upload"]
    assert "9 files of Panache" in capsys.readouterr().out
