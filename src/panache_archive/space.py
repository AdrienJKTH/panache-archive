"""The Hugging Face Space that computes views at ~100 m (Panache's D74).

Panache's button "compute in high definition" needs a machine that is not Adrien's, and a free
Space is one. But a public Space's image can be pulled by anyone, and Panache's repository is
private: only the computation goes there (Adrien's choice, 2026-11-06). So the Space's files are
worked out, not listed: `panache.compute_app`, `panache.fetch_archive` and what they import, and
nothing else of Panache's code.
"""

import ast
import logging
import shutil
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Final, Protocol

logger = logging.getLogger(__name__)

REPOSITORY: Final = "adrien4916/panache-compute"
ROOTS: Final = ("panache.compute_app", "panache.fetch_archive")
PACKAGE: Final = "panache"


class Hub(Protocol):
    def create_repo(
        self, repo_id: str, *, repo_type: str, space_sdk: str, exist_ok: bool
    ) -> object: ...

    def upload_folder(
        self,
        *,
        repo_id: str,
        folder_path: Path,
        repo_type: str,
        commit_message: str,
        delete_patterns: str,
    ) -> object: ...


def imported_files(source: Path, roots: Sequence[str] = ROOTS) -> list[Path]:
    """The files of `source` that `roots` need, each module's packages included, relative to
    `source`, sorted."""
    found: dict[str, Path] = {}
    waiting = list(roots)
    while waiting:
        module = waiting.pop()
        path = _module_file(source, module)
        if module in found or path is None:
            continue
        found[module] = path
        waiting.extend(_parents(module))
        waiting.extend(_imports(path))
    missing = [root for root in roots if root not in found]
    if missing:
        raise ValueError(f"{missing} not in {source}")
    return sorted(path.relative_to(source) for path in found.values())


def bundle(source: Path, space: Path, requirements: Path, out: Path) -> list[Path]:
    """The Space's folder in `out`: the files of `space` (its Dockerfile and card), the pinned
    `requirements`, and the code `imported_files` finds."""
    shutil.copytree(space, out, dirs_exist_ok=True)
    shutil.copy(requirements, out / "requirements.txt")
    files = imported_files(source)
    for relative in files:
        target = out / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(source / relative, target)
    return files


def deploy(hub: Hub, folder: Path, repository: str = REPOSITORY) -> None:
    """Make the Space hold exactly `folder`, creating it the first time."""
    hub.create_repo(repository, repo_type="space", space_sdk="docker", exist_ok=True)
    hub.upload_folder(
        repo_id=repository,
        folder_path=folder,
        repo_type="space",
        commit_message="Deploy the computation of views at ~100 m",
        delete_patterns="*",
    )


def _module_file(source: Path, module: str) -> Path | None:
    base = source.joinpath(*module.split("."))
    for path in (base.with_suffix(".py"), base / "__init__.py"):
        if path.exists():
            return path
    return None


def _parents(module: str) -> Iterable[str]:
    parts = module.split(".")
    return (".".join(parts[:i]) for i in range(1, len(parts)))


def _imports(path: Path) -> Iterable[str]:
    """The Panache modules a file names: for `from a import b`, `b` may be one too."""
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            yield from (a.name for a in node.names if _is_ours(a.name))
        elif isinstance(node, ast.ImportFrom) and node.module and _is_ours(node.module):
            yield node.module
            yield from (f"{node.module}.{a.name}" for a in node.names)


def _is_ours(module: str) -> bool:
    return module == PACKAGE or module.startswith(f"{PACKAGE}.")
