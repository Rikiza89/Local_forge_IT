"""
Filesystem adapter — file read/write and directory operations using pathlib.Path.
Implements the FileSystemPort interface.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from localforge.domain.exceptions import FileWriteError
from localforge.domain.models import FileNode, FileStatus

logger = logging.getLogger(__name__)

_DEFAULT_IGNORE_DIRS = frozenset({
    ".git", ".localforge", "__pycache__", ".venv", "venv",
    "node_modules", ".tox", "dist", "build", ".mypy_cache",
    ".pytest_cache", ".ruff_cache", "*.egg-info",
})

CODE_EXTENSIONS = frozenset({
    ".py", ".js", ".ts", ".jsx", ".tsx", ".go", ".rs", ".java",
    ".html", ".css", ".scss", ".sass", ".vue", ".svelte",
    ".c", ".cpp", ".h", ".hpp", ".cs", ".rb", ".php",
    ".sh", ".bash", ".zsh", ".yaml", ".yml", ".toml", ".json",
    ".md", ".rst", ".txt", ".sql", ".graphql", ".proto",
    ".dockerfile", ".tf", ".hcl",
})


class FileSystemAdapter:

    def read_text(self, path: Path) -> str:
        return path.read_text(encoding="utf-8", errors="replace")

    def write_text(self, path: Path, content: str) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            logger.debug("File written: %s", path)
        except OSError as exc:
            raise FileWriteError(f"Failed to write file: {path} — {exc}") from exc

    def list_files(self, root: Path, extensions: Optional[Iterable[str]] = None, ignore_dirs: Optional[Iterable[str]] = None) -> List[Path]:
        ext_set = frozenset(extensions) if extensions else None
        skip_dirs = _DEFAULT_IGNORE_DIRS | (frozenset(ignore_dirs) if ignore_dirs else frozenset())
        result: List[Path] = []
        self._walk(root, root, ext_set, skip_dirs, result)
        return sorted(result)

    def _walk(self, base: Path, current: Path, ext_set: Optional[frozenset], skip_dirs: frozenset, result: List[Path]) -> None:
        try:
            entries = sorted(current.iterdir(), key=lambda p: (p.is_file(), p.name))
        except PermissionError:
            logger.warning("Permission denied: %s", current)
            return
        for entry in entries:
            if entry.is_dir():
                if entry.name not in skip_dirs:
                    self._walk(base, entry, ext_set, skip_dirs, result)
            elif entry.is_file():
                if ext_set is None or entry.suffix.lower() in ext_set:
                    result.append(entry)

    def list_code_files(self, root: Path) -> List[Path]:
        return self.list_files(root, extensions=CODE_EXTENSIONS)

    def has_code_files(self, root: Path) -> bool:
        for _ in self._iter_code_files_limit(root, limit=1):
            return True
        return False

    def _iter_code_files_limit(self, root: Path, limit: int) -> Iterable[Path]:
        count = 0
        for path in self.list_code_files(root):
            yield path
            count += 1
            if count >= limit:
                break

    def build_file_tree(self, root: Path) -> List[FileNode]:
        return self._build_nodes(root, root)

    def _build_nodes(self, base: Path, current: Path) -> List[FileNode]:
        nodes: List[FileNode] = []
        try:
            entries = sorted(current.iterdir(), key=lambda p: (p.is_file(), p.name))
        except PermissionError:
            return nodes
        for entry in entries:
            rel = str(entry.relative_to(base))
            if entry.is_dir():
                if entry.name.startswith(".") and entry.name not in {".localforge"}:
                    continue
                if entry.name in _DEFAULT_IGNORE_DIRS:
                    continue
                children = self._build_nodes(base, entry)
                nodes.append(FileNode(name=entry.name, path=rel, is_dir=True, children=children))
            elif entry.is_file():
                try:
                    stat = entry.stat()
                    nodes.append(FileNode(name=entry.name, path=rel, is_dir=False, size=stat.st_size, modified_at=stat.st_mtime))
                except OSError:
                    nodes.append(FileNode(name=entry.name, path=rel, is_dir=False))
        return nodes

    def get_mtime_size(self, path: Path) -> Tuple[float, int]:
        stat = path.stat()
        return stat.st_mtime, stat.st_size

    def exists(self, path: Path) -> bool:
        return path.exists()

    def read_lines_range(self, path: Path, start: int, end: int) -> List[str]:
        lines = self.read_text(path).splitlines()
        return lines[start:end]

    def count_lines(self, path: Path) -> int:
        return len(self.read_text(path).splitlines())
