"""
Git adapter — git operation wrapper using subprocess.
Implements the GitPort interface.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import List

from localforge.domain.exceptions import GitOperationError

logger = logging.getLogger(__name__)


class GitAdapter:

    def _run(self, args: List[str], cwd: Path) -> str:
        cmd = ["git"] + args
        try:
            result = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, check=True, encoding="utf-8", errors="replace")
            return result.stdout
        except subprocess.CalledProcessError as exc:
            raise GitOperationError(f"git command failed: {' '.join(cmd)}\n{exc.stderr}") from exc
        except FileNotFoundError as exc:
            raise GitOperationError("git command not found") from exc

    def _run_no_raise(self, args: List[str], cwd: Path, default: str = "") -> str:
        try:
            return self._run(args, cwd)
        except GitOperationError:
            return default

    def init(self, path: Path) -> None:
        try:
            self._run(["init"], path)
            logger.info("git init complete: %s", path)
            self._run_no_raise(["config", "commit.gpgsign", "false"], path)
            gitignore_path = path / ".gitignore"
            if not gitignore_path.exists():
                gitignore_path.write_text(".localforge/app.log\n__pycache__/\n*.pyc\n.venv/\nvenv/\n", encoding="utf-8")
        except GitOperationError as exc:
            raise GitOperationError(f"git init failed: {exc}") from exc

    def commit_all(self, path: Path, message: str) -> str:
        try:
            self._run_no_raise(["config", "user.email", "localforge@local"], path)
            self._run_no_raise(["config", "user.name", "LocalForge"], path)
            self._run(["add", "-A"], path)
            try:
                self._run(["-c", "commit.gpgsign=false", "commit", "-m", message], path)
            except GitOperationError as exc:
                if "nothing to commit" in str(exc) or "nothing added to commit" in str(exc):
                    logger.debug("Nothing to commit: %s", path)
                    return ""
                raise
            commit_hash = self._run_no_raise(["rev-parse", "--short", "HEAD"], path).strip()
            logger.info("Commit complete: %s (%s)", message[:50], commit_hash)
            return commit_hash
        except GitOperationError:
            raise

    def get_log(self, path: Path, max_entries: int = 20) -> List[dict]:
        if not self._is_git_repo(path):
            return []
        try:
            output = self._run(["log", f"-{max_entries}", "--pretty=format:%H|%s|%an|%ai", "--no-merges"], path)
        except GitOperationError:
            return []
        entries: List[dict] = []
        for line in output.strip().splitlines():
            parts = line.split("|", 3)
            if len(parts) == 4:
                entries.append({"hash": parts[0][:8], "message": parts[1], "author": parts[2], "date": parts[3]})
        return entries

    def get_diff(self, path: Path) -> str:
        if not self._is_git_repo(path):
            return ""
        return self._run_no_raise(["diff", "HEAD"], path)

    def get_status(self, path: Path) -> str:
        if not self._is_git_repo(path):
            return ""
        return self._run_no_raise(["status", "--short"], path)

    def get_current_branch(self, path: Path) -> str:
        if not self._is_git_repo(path):
            return ""
        return self._run_no_raise(["rev-parse", "--abbrev-ref", "HEAD"], path, default="").strip()

    def _is_git_repo(self, path: Path) -> bool:
        return (path / ".git").is_dir()
