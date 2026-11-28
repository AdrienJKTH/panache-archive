"""Panache's private Hugging Face dataset: the trained models and the experiments' results.

The archive's data are open (CAMS, ERA5, MERRA-2, winds, layers), and stay in public
releases mirrored on the public dataset `adrien4916/air_qual_indicators`. What Panache learns from
them is not: the LightGBM boosters (their text files carry the feature names, so they show the
feature engineering too), the experiments' tables and scores, and the maps Panache computes
go to the private dataset
`adrien4916/panache-models`, one folder per release name (`forecast-models-v1`,
`downscaled-v1`, ...), where only the workflows that build them (with `HF_TOKEN`, a write token)
and Panache's backend (with a read token) reach them.

`Mirror` leaves them out: `is_private` says which releases and assets never go to the public
dataset.
"""

import fnmatch
import logging
import re
import shutil
import tempfile
import time
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any, Final, Protocol

from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi
from huggingface_hub.errors import EntryNotFoundError, HfHubHTTPError, RepositoryNotFoundError
from huggingface_hub.hf_api import RepoFile

REPO_ID: Final = "adrien4916/panache-models"
PRIVATE_RELEASES: Final = frozenset(
    {
        "forecast-models-v1",
        "forecast-experiment-v1",
        "downscaled-v1",
        "downscaled-europe-v1",
        "downscaled-towns-v1",
        "maps-europe-v1",
        "maps-live-v1",
        "maps-analysed-v1",
        "seasons-europe-v1",
        "innovations-live-v1",
    }
)
"""Releases that are private whole: what Panache computes (the models, the experiments, the maps),
never published on GitHub nor mirrored on the public dataset. The others are open data,
repackaged."""
PRIVATE_ASSET: Final = re.compile(r"\.lgb\.txt$")
"""A model's file, in whichever release it is (`downscaled-v1`, `downscaled-towns-v1`, ...)."""

ATTEMPTS: Final = 4
RETRY_PAUSE: Final = 5.0
CONFLICTS: Final = frozenset({409, 412, 429})
"""Answers to a commit that another commit, made at the same time by a sibling job of a matrix,
may cause: it is made again (and so is a server error)."""

logger = logging.getLogger(__name__)


def is_private(release: str, asset: str | None = None) -> bool:
    """Whether a release, or an asset of it, stays out of every public host."""
    if release in PRIVATE_RELEASES:
        return True
    return asset is not None and PRIVATE_ASSET.search(asset) is not None


class Api(Protocol):
    """What the store needs of `huggingface_hub.HfApi` (or a fake)."""

    def create_repo(
        self, repo_id: str, *, repo_type: str, private: bool, exist_ok: bool
    ) -> Any: ...

    def repo_info(self, repo_id: str, *, repo_type: str) -> Any: ...

    def list_repo_tree(
        self, repo_id: str, *, path_in_repo: str, repo_type: str
    ) -> Iterable[Any]: ...

    def create_commit(
        self,
        repo_id: str,
        operations: Sequence[CommitOperationAdd | CommitOperationDelete],
        *,
        commit_message: str,
        repo_type: str,
    ) -> Any: ...

    def super_squash_history(self, repo_id: str, *, repo_type: str) -> Any: ...

    def dataset_info(self, repo_id: str, *, expand: Any) -> Any: ...

    def hf_hub_download(
        self, repo_id: str, filename: str, *, repo_type: str, local_dir: str
    ) -> str: ...


class PublicApi(Protocol):
    """What `purge_public` needs of `huggingface_hub.HfApi` (or a fake)."""

    def list_repo_tree(self, repo_id: str, *, repo_type: str, recursive: bool) -> Iterable[Any]: ...

    def create_commit(
        self,
        repo_id: str,
        operations: Sequence[CommitOperationDelete],
        *,
        commit_message: str,
        repo_type: str,
    ) -> Any: ...

    def super_squash_history(self, repo_id: str, *, repo_type: str) -> Any: ...


def purge_public(api: PublicApi, repo_id: str, *, squash: bool) -> list[str]:
    """Deletes from the public dataset `repo_id` (one folder per release) every file that
    `is_private` says never belonged there, and returns their paths. With `squash`, its history
    is squashed into one commit afterwards, so that no old revision keeps them (irreversible)."""
    if repo_id == REPO_ID:
        raise ValueError(f"{repo_id} is the private dataset")
    leaks = sorted(
        entry.path
        for entry in api.list_repo_tree(repo_id, repo_type="dataset", recursive=True)
        if isinstance(entry, RepoFile)
        and "/" in entry.path
        and is_private(entry.path.split("/", 1)[0], entry.path.rsplit("/", 1)[1])
    )
    if leaks:
        api.create_commit(
            repo_id,
            [CommitOperationDelete(path_in_repo=path) for path in leaks],
            commit_message=f"Remove {len(leaks)} private files",
            repo_type="dataset",
        )
    if squash:
        api.super_squash_history(repo_id, repo_type="dataset")
    return leaks


class PrivateStore:
    """The private dataset, by folder. It never writes to a repository that is public."""

    def __init__(self, token: str, repo_id: str = REPO_ID, api: Api | None = None) -> None:
        self._api: Api = api or HfApi(token=token)
        self._repo_id = repo_id

    def files(self, folder: str) -> dict[str, int]:
        """The sizes of the files of `folder`, by name: none when it (or the dataset) does not
        exist yet."""
        try:
            entries = self._api.list_repo_tree(
                self._repo_id, path_in_repo=folder, repo_type="dataset"
            )
            return {e.path.rsplit("/", 1)[-1]: e.size for e in entries if isinstance(e, RepoFile)}
        except (EntryNotFoundError, RepositoryNotFoundError):
            return {}

    def used(self) -> int:
        """The bytes the dataset takes of the private quota, old revisions included, as
        Hugging Face counts them; an error if it does not say."""
        info = self._api.dataset_info(self._repo_id, expand=["usedStorage"])
        if info.used_storage is None:
            raise ValueError(f"Hugging Face does not say what {self._repo_id} takes")
        return int(info.used_storage)

    def names(self, folder: str) -> list[str]:
        """The files of `folder`, by name."""
        return sorted(self.files(folder))

    def upload(
        self, folder: str, paths: Sequence[Path], *, sleep: Callable[[float], None] = time.sleep
    ) -> None:
        """`paths` into `folder`, in one commit; the dataset is created private when missing. A
        directory goes with the files under it, at their paths from it."""
        self._create()
        operations = [
            CommitOperationAdd(path_in_repo=f"{folder}/{name}", path_or_fileobj=str(path))
            for name, path in _files(paths)
        ]
        names = ", ".join(op.path_in_repo.rsplit("/", 1)[-1] for op in operations)
        self._commit(operations, f"{folder}: {names}", sleep)
        logger.info("%d files in %s/%s", len(operations), self._repo_id, folder)

    def delete(
        self,
        folder: str,
        names: Sequence[str] | None = None,
        *,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        """`names` out of `folder`, or the whole folder without names, in one commit."""
        self._create()
        operations = (
            [CommitOperationDelete(path_in_repo=f"{folder}/", is_folder=True)]
            if names is None
            else [CommitOperationDelete(path_in_repo=f"{folder}/{name}") for name in names]
        )
        if names is None or names:
            self._commit(operations, f"Delete {folder}/{', '.join(names or ['*'])}", sleep)

    def squash(self) -> None:
        """Squashes the dataset's history into one commit, so that the files it dropped no longer
        count against the free quota. Irreversible, and not to run while another job commits."""
        self._create()
        self._api.super_squash_history(self._repo_id, repo_type="dataset")

    def download(self, folder: str, pattern: str, out: Path) -> list[Path]:
        """The files of `folder` whose name matches `pattern` (a glob), written into `out`."""
        out.mkdir(parents=True, exist_ok=True)
        found = []
        for name in fnmatch.filter(self.names(folder), pattern):
            with tempfile.TemporaryDirectory() as scratch:
                cached = self._api.hf_hub_download(
                    self._repo_id,
                    filename=f"{folder}/{name}",
                    repo_type="dataset",
                    local_dir=scratch,
                )
                shutil.copyfile(cached, out / name)
            found.append(out / name)
        return found

    def _create(self) -> None:
        self._api.create_repo(self._repo_id, repo_type="dataset", private=True, exist_ok=True)
        if not self._api.repo_info(self._repo_id, repo_type="dataset").private:
            raise PermissionError(f"{self._repo_id} is public: nothing is written to it")

    def _commit(
        self,
        operations: Sequence[CommitOperationAdd | CommitOperationDelete],
        message: str,
        sleep: Callable[[float], None],
    ) -> None:
        for attempt in range(1, ATTEMPTS + 1):
            try:
                self._api.create_commit(
                    self._repo_id, operations, commit_message=message, repo_type="dataset"
                )
                return
            except HfHubHTTPError as error:
                status = error.response.status_code
                if attempt == ATTEMPTS or not (status in CONFLICTS or status >= 500):
                    raise
                logger.warning("HTTP %d, again in %.0f s: %s", status, RETRY_PAUSE, message[:80])
                sleep(RETRY_PAUSE * attempt)


def _files(paths: Sequence[Path]) -> list[tuple[str, Path]]:
    """Each file to upload with its name in the folder: a file by its name, a directory's files
    by their paths from it."""
    found: list[tuple[str, Path]] = []
    for path in paths:
        if path.is_dir():
            found += [
                (child.relative_to(path).as_posix(), child)
                for child in sorted(path.rglob("*"))
                if child.is_file()
            ]
        else:
            found.append((path.name, path))
    return found
