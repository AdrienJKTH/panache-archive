"""Panache's private Hugging Face dataset: the trained models and the experiments' results.

The archive's data are open (CAMS, ERA5, MERRA-2, winds, layers, the maps), and stay in public
releases mirrored on the public dataset `adrien4916/air_qual_indicators`. What Panache learns from
them is not: the LightGBM boosters (their text files carry the feature names, so they show the
feature engineering too) and the experiments' tables and scores go to the private dataset
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

from huggingface_hub import CommitOperationAdd, HfApi
from huggingface_hub.errors import EntryNotFoundError, HfHubHTTPError, RepositoryNotFoundError
from huggingface_hub.hf_api import RepoFile

REPO_ID: Final = "adrien4916/panache-models"
PRIVATE_RELEASES: Final = frozenset({"forecast-models-v1", "forecast-experiment-v1"})
"""Releases that are private whole: they are no longer published on GitHub."""
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
        operations: Sequence[CommitOperationAdd],
        *,
        commit_message: str,
        repo_type: str,
    ) -> Any: ...

    def hf_hub_download(
        self, repo_id: str, filename: str, *, repo_type: str, local_dir: str
    ) -> str: ...


class PrivateStore:
    """The private dataset, by folder. It never writes to a repository that is public."""

    def __init__(self, token: str, repo_id: str = REPO_ID, api: Api | None = None) -> None:
        self._api: Api = api or HfApi(token=token)
        self._repo_id = repo_id

    def names(self, folder: str) -> list[str]:
        """The files of `folder`, none when it (or the dataset) does not exist yet."""
        try:
            entries = self._api.list_repo_tree(
                self._repo_id, path_in_repo=folder, repo_type="dataset"
            )
            return sorted(e.path.rsplit("/", 1)[-1] for e in entries if isinstance(e, RepoFile))
        except (EntryNotFoundError, RepositoryNotFoundError):
            return []

    def upload(
        self, folder: str, files: Sequence[Path], *, sleep: Callable[[float], None] = time.sleep
    ) -> None:
        """`files` into `folder`, in one commit; the dataset is created private when missing."""
        self._api.create_repo(self._repo_id, repo_type="dataset", private=True, exist_ok=True)
        if not self._api.repo_info(self._repo_id, repo_type="dataset").private:
            raise PermissionError(f"{self._repo_id} is public: nothing is written to it")
        operations = [
            CommitOperationAdd(path_in_repo=f"{folder}/{path.name}", path_or_fileobj=str(path))
            for path in files
        ]
        message = f"{folder}: {', '.join(path.name for path in files)}"
        for attempt in range(1, ATTEMPTS + 1):
            try:
                self._api.create_commit(
                    self._repo_id, operations, commit_message=message, repo_type="dataset"
                )
                break
            except HfHubHTTPError as error:
                status = error.response.status_code
                if attempt == ATTEMPTS or not (status in CONFLICTS or status >= 500):
                    raise
                logger.warning("%s: HTTP %d, again in %.0f s", folder, status, RETRY_PAUSE)
                sleep(RETRY_PAUSE * attempt)
        logger.info("%d files in %s/%s", len(files), self._repo_id, folder)

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
