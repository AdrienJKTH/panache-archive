from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
from huggingface_hub import CommitOperationAdd, CommitOperationDelete
from huggingface_hub.errors import HfHubHTTPError, RepositoryNotFoundError
from huggingface_hub.hf_api import RepoFile

from panache_archive import cli, mirror, private
from panache_archive.mirror import Asset, git_blob_id


class FakeApi:
    """The private dataset's files by path, and the calls made to it."""

    def __init__(self, files: dict[str, bytes] | None = None, *, public: bool = False) -> None:
        self.files = files or {}
        self.public = public
        self.created: list[dict[str, Any]] = []
        self.commits: list[tuple[str, list[Any], str]] = []
        self.squashed = 0

    def create_repo(self, repo_id: str, *, repo_type: str, private: bool, exist_ok: bool) -> None:
        self.created.append(
            {"repo_id": repo_id, "repo_type": repo_type, "private": private, "exist_ok": exist_ok}
        )

    def repo_info(self, repo_id: str, *, repo_type: str) -> Any:
        assert (repo_id, repo_type) == (private.REPO_ID, "dataset")

        class Info:
            private = not self.public

        return Info()

    def list_repo_tree(self, repo_id: str, *, path_in_repo: str, repo_type: str) -> list[object]:
        assert (repo_id, repo_type) == (private.REPO_ID, "dataset")
        if not self.files:
            raise RepositoryNotFoundError("missing", response=Mock(headers={}))
        return [
            RepoFile(path=path, size=len(content), oid="x")  # type: ignore[no-untyped-call]
            for path, content in sorted(self.files.items())
            if path.startswith(f"{path_in_repo}/")
        ]

    def create_commit(
        self,
        repo_id: str,
        operations: Sequence[CommitOperationAdd | CommitOperationDelete],
        *,
        commit_message: str,
        repo_type: str,
    ) -> None:
        assert repo_type == "dataset"
        self.commits.append((repo_id, list(operations), commit_message))

    def super_squash_history(self, repo_id: str, *, repo_type: str) -> None:
        assert (repo_id, repo_type) == (private.REPO_ID, "dataset")
        self.squashed += 1

    taken: int | None = 82_786_001_590

    def dataset_info(self, repo_id: str, *, expand: list[str]) -> SimpleNamespace:
        assert (repo_id, expand) == (private.REPO_ID, ["usedStorage"])
        return SimpleNamespace(used_storage=self.taken)

    def hf_hub_download(
        self,
        repo_id: str,
        filename: str,
        *,
        repo_type: str,
        local_dir: str,
    ) -> str:
        assert (repo_id, repo_type) == (private.REPO_ID, "dataset")
        target = Path(local_dir) / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(self.files[filename])
        return str(target)


def _store(api: FakeApi) -> private.PrivateStore:
    return private.PrivateStore("token", api=api)


def test_the_models_and_the_experiments_are_private() -> None:
    assert private.is_private("forecast-models-v1")
    assert private.is_private("forecast-experiment-v1", "forecast-scores.md")
    assert private.is_private("downscaled-europe-v1", "downscaling-ozone.lgb.txt")
    assert private.is_private("downscaled-towns-v1", "downscaling-fine-nitrogen_dioxide.lgb.txt")
    assert private.is_private("downscaled-europe-v1", "downscaled-ozone-2021.days")
    assert private.is_private("maps-live-v1", "live.json")
    assert private.is_private("seasons-europe-v1")
    assert not private.is_private("downscaling-europe-v1", "static-europe-1km.npz")
    assert not private.is_private("downscaling-europe-v1", "stations-FR-pm10.parquet")
    assert not private.is_private("stations-live-v1")
    assert not private.is_private("cams-europe-v1", "cams-pm10-2021-validated.cells")


def test_the_mirror_leaves_the_models_out_and_removes_those_it_holds() -> None:
    maps = Asset("static-europe-1km.npz", 9, "d")
    model = Asset("downscaling-ozone.lgb.txt", 3, "m")

    todo = mirror.plan([maps, model], [model], b"card", git_blob_id(b"card"))

    assert [a.name for a in todo.upload] == ["static-europe-1km.npz"]
    assert todo.delete == ("downscaling-ozone.lgb.txt",)


def test_a_private_release_is_not_mirrored(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("HF_TOKEN", "token")

    code = cli.main(
        ["mirror", "--release", "r.json", "--card", "c.md", "--folder", "forecast-models-v1"]
    )

    assert code == 0
    assert "private" in capsys.readouterr().out


def test_names_lists_a_folder_and_none_when_the_dataset_does_not_exist() -> None:
    api = FakeApi({"a-v1/x.txt": b"1", "a-v1/y.txt": b"22", "b-v1/z.txt": b"3"})

    assert _store(api).names("a-v1") == ["x.txt", "y.txt"]
    assert _store(FakeApi()).names("a-v1") == []


def test_upload_creates_the_dataset_private_and_commits_the_files_once(tmp_path: Path) -> None:
    first, second = tmp_path / "a.txt", tmp_path / "b.txt"
    first.write_bytes(b"1")
    second.write_bytes(b"2")
    api = FakeApi()

    _store(api).upload("a-v1", [first, second])

    assert api.created == [
        {"repo_id": private.REPO_ID, "repo_type": "dataset", "private": True, "exist_ok": True}
    ]
    ((repo, operations, message),) = api.commits
    assert repo == private.REPO_ID
    assert [o.path_in_repo for o in operations] == ["a-v1/a.txt", "a-v1/b.txt"]
    assert message == "a-v1: a.txt, b.txt"


def test_upload_never_writes_to_a_public_dataset(tmp_path: Path) -> None:
    file = tmp_path / "a.txt"
    file.write_bytes(b"1")
    api = FakeApi(public=True)

    with pytest.raises(PermissionError, match="public"):
        _store(api).upload("a-v1", [file])

    assert api.commits == []


def test_download_writes_the_files_that_match(tmp_path: Path) -> None:
    api = FakeApi({"a-v1/x.lgb.txt": b"model", "a-v1/y.json": b"{}"})

    found = _store(api).download("a-v1", "*.lgb.txt", tmp_path / "out")

    assert found == [tmp_path / "out" / "x.lgb.txt"]
    assert found[0].read_bytes() == b"model"
    assert not (tmp_path / "out" / "y.json").exists()


def test_the_commands_need_a_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)

    with pytest.raises(SystemExit, match="HF_TOKEN"):
        cli.main(["private", "list", "--folder", "a-v1"])


def test_the_commands_list_upload_and_download(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    api = FakeApi({"a-v1/x.lgb.txt": b"model"})
    monkeypatch.setenv("HF_TOKEN", "token")
    monkeypatch.setattr(private, "HfApi", lambda **_: api)
    file = tmp_path / "up.txt"
    file.write_bytes(b"1")

    assert cli.main(["private", "list", "--folder", "a-v1"]) == 0
    assert capsys.readouterr().out == "x.lgb.txt\n"
    assert cli.main(["private", "upload", "--folder", "a-v1", str(file)]) == 0
    assert len(api.commits) == 1
    assert cli.main(["private", "download", "--folder", "a-v1", "--dir", str(tmp_path / "d")]) == 0
    assert (tmp_path / "d" / "x.lgb.txt").read_bytes() == b"model"
    nothing = ["private", "download", "--folder", "a-v1", "--pattern", "*.zip"]
    nothing += ["--dir", str(tmp_path)]
    assert cli.main(nothing) == 1
    assert cli.main([*nothing, "--allow-empty"]) == 0


def _failing(api: FakeApi, statuses: list[int]) -> None:
    commit = api.create_commit

    def create_commit(*args: Any, **kwargs: Any) -> None:
        if statuses:
            raise HfHubHTTPError("busy", response=Mock(headers={}, status_code=statuses.pop(0)))
        commit(*args, **kwargs)

    api.create_commit = create_commit  # type: ignore[method-assign]


def test_a_commit_that_met_another_is_made_again(tmp_path: Path) -> None:
    file = tmp_path / "a.txt"
    file.write_bytes(b"1")
    api = FakeApi()
    _failing(api, [412, 503])
    waits: list[float] = []

    _store(api).upload("a-v1", [file], sleep=waits.append)

    assert len(api.commits) == 1
    assert waits == [5.0, 10.0]


@pytest.mark.parametrize(("statuses", "raised"), [([401], 401), ([409, 409, 409, 409], 409)])
def test_a_refusal_or_too_many_conflicts_is_raised(
    tmp_path: Path, statuses: list[int], raised: int
) -> None:
    file = tmp_path / "a.txt"
    file.write_bytes(b"1")
    api = FakeApi()
    _failing(api, statuses)

    with pytest.raises(HfHubHTTPError) as error:
        _store(api).upload("a-v1", [file], sleep=lambda _: None)

    assert error.value.response.status_code == raised
    assert api.commits == []


def test_download_may_find_nothing(tmp_path: Path) -> None:
    api = FakeApi({"a-v1/x.txt": b"1"})

    assert _store(api).download("a-v1", "*.parquet", tmp_path) == []


class FakePublicApi:
    def __init__(self, paths: list[str]) -> None:
        self.paths = paths
        self.deleted: list[str] = []
        self.squashed = 0

    def list_repo_tree(self, repo_id: str, *, repo_type: str, recursive: bool) -> list[object]:
        assert (repo_id, repo_type, recursive) == ("public/dataset", "dataset", True)
        return [
            RepoFile(path=path, size=1, oid="x")  # type: ignore[no-untyped-call]
            for path in self.paths
        ]

    def create_commit(
        self, repo_id: str, operations: Sequence[Any], *, commit_message: str, repo_type: str
    ) -> None:
        assert (repo_id, repo_type) == ("public/dataset", "dataset")
        self.deleted += [o.path_in_repo for o in operations]
        assert commit_message == f"Remove {len(operations)} private files"

    def super_squash_history(self, repo_id: str, *, repo_type: str) -> None:
        assert (repo_id, repo_type) == ("public/dataset", "dataset")
        self.squashed += 1


PUBLIC = [
    "README.md",
    "cams-europe-v1/cams-pm10-2021-validated.cells",
    "downscaling-europe-v1/static-europe-1km.npz",
    "downscaled-europe-v1/downscaling-ozone.lgb.txt",
    "downscaled-towns-v1/downscaling-fine-nitrogen_dioxide.lgb.txt",
    "forecast-experiment-v1/forecast-scores.md",
    "forecast-models-v1/lightgbm-j1.txt",
]


def test_purge_removes_the_private_files_and_squashes_the_history() -> None:
    api = FakePublicApi(PUBLIC)

    removed = private.purge_public(api, "public/dataset", squash=True)

    assert (
        removed
        == api.deleted
        == [
            "downscaled-europe-v1/downscaling-ozone.lgb.txt",
            "downscaled-towns-v1/downscaling-fine-nitrogen_dioxide.lgb.txt",
            "forecast-experiment-v1/forecast-scores.md",
            "forecast-models-v1/lightgbm-j1.txt",
        ]
    )
    assert api.squashed == 1


def test_purge_without_leaks_commits_nothing_and_keeps_the_history_unless_asked() -> None:
    api = FakePublicApi(PUBLIC[:3])

    assert private.purge_public(api, "public/dataset", squash=False) == []
    assert (api.deleted, api.squashed) == ([], 0)


def test_purge_never_touches_the_private_dataset() -> None:
    with pytest.raises(ValueError, match="private"):
        private.purge_public(FakePublicApi([]), private.REPO_ID, squash=True)


def test_the_purge_command_reports_what_it_removed(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    api = FakePublicApi(PUBLIC)
    api_class = lambda **_: api  # noqa: E731
    monkeypatch.setenv("HF_TOKEN", "token")
    monkeypatch.setattr(cli, "HfApi", api_class)
    monkeypatch.setattr(mirror, "REPO_ID", "public/dataset")

    assert cli.main(["private", "purge-public", "--squash"]) == 0

    out = capsys.readouterr().out
    assert "4 private files removed from public/dataset" in out
    assert "forecast-models-v1/lightgbm-j1.txt" in out
    assert "history squashed" in out
    monkeypatch.delenv("HF_TOKEN")
    with pytest.raises(SystemExit, match="HF_TOKEN"):
        cli.main(["private", "purge-public"])


def test_a_directory_is_uploaded_with_the_paths_of_its_files(tmp_path: Path) -> None:
    band = tmp_path / "band-3"
    (band / "sub").mkdir(parents=True)
    (band / "a.days").write_bytes(b"a")
    (band / "sub" / "b.npz").write_bytes(b"b")
    loose = tmp_path / "c.json"
    loose.write_bytes(b"c")
    api = FakeApi()

    _store(api).upload("scratch/7", [band, loose])

    ((_, operations, _),) = api.commits
    assert [o.path_in_repo for o in operations] == [
        "scratch/7/a.days",
        "scratch/7/sub/b.npz",
        "scratch/7/c.json",
    ]


def test_files_report_their_sizes() -> None:
    api = FakeApi({"a-v1/x.txt": b"1", "a-v1/y.txt": b"22"})

    assert _store(api).files("a-v1") == {"x.txt": 1, "y.txt": 2}


def test_delete_removes_files_or_a_whole_folder() -> None:
    api = FakeApi()

    _store(api).delete("a-v1", ["x.txt", "y.txt"])
    _store(api).delete("scratch/7")
    _store(api).delete("a-v1", [])  # nothing to delete: no commit

    first, second = api.commits
    assert [(o.path_in_repo, o.is_folder) for o in first[1]] == [
        ("a-v1/x.txt", False),
        ("a-v1/y.txt", False),
    ]
    assert [(o.path_in_repo, o.is_folder) for o in second[1]] == [("scratch/7/", True)]
    assert len(api.commits) == 2


def test_squash_squashes_the_history_of_a_private_dataset_only() -> None:
    api = FakeApi()
    _store(api).squash()
    assert api.squashed == 1

    with pytest.raises(PermissionError):
        _store(FakeApi(public=True)).squash()


def test_the_delete_and_squash_commands(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    api = FakeApi({"a-v1/x.txt": b"1"})
    monkeypatch.setenv("HF_TOKEN", "token")
    monkeypatch.setattr(private, "HfApi", lambda **_: api)

    assert cli.main(["private", "delete", "--folder", "a-v1", "x.txt"]) == 0
    assert cli.main(["private", "delete", "--folder", "scratch/7"]) == 0
    assert cli.main(["private", "squash"]) == 0

    assert len(api.commits) == 2
    assert api.squashed == 1
    out = capsys.readouterr().out
    assert "x.txt out of a-v1" in out
    assert "the whole folder out of scratch/7" in out


def test_what_the_dataset_takes_is_hugging_face_s_count() -> None:
    api = FakeApi()
    store = private.PrivateStore("token", api=api)

    assert store.used() == 82_786_001_590
    api.taken = None
    with pytest.raises(ValueError, match="does not say"):
        store.used()
