"""
Project service — project management, mode detection, and state management.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from localforge.domain.models import (
    FileNode, GenerationLogEntry, GenerationPlan,
    Project, ProjectConfig, ProjectMode, ResumeState,
)
from localforge.infrastructure.filesystem_adapter import FileSystemAdapter
from localforge.infrastructure.git_adapter import GitAdapter
from localforge.infrastructure.index_adapter import IndexAdapter

logger = logging.getLogger(__name__)

_LOCALFORGE_DIR = ".localforge"
_CONFIG_FILE = "config.json"
_CONTEXT_FILE = "context.md"
_INDEX_JSONL = "index.jsonl"
_GENERATION_LOG = "generation_log.jsonl"
_PROJECT_INDEX = "project_index.json"
_APP_LOG = "app.log"


class ProjectService:

    def __init__(self, fs: FileSystemAdapter, git: GitAdapter, index: IndexAdapter) -> None:
        self._fs = fs
        self._git = git
        self._index = index
        self._current_project: Optional[Project] = None

    @property
    def current_project(self) -> Optional[Project]:
        return self._current_project

    def open_project(self, root: Path) -> Project:
        mode = self.detect_project_mode(root)
        config = self._load_or_create_config(root, mode)
        file_tree = self._fs.build_file_tree(root)
        resume_state: Optional[ResumeState] = None
        if mode == ProjectMode.RESUME:
            resume_state = self._build_resume_state(root)
        project = Project(root=root, mode=mode, config=config, file_tree=file_tree, resume_state=resume_state)
        self._current_project = project
        logger.info("Project opened: %s (mode=%s)", root, mode.value)
        return project

    def detect_project_mode(self, root: Path) -> ProjectMode:
        lf_dir = root / _LOCALFORGE_DIR
        config_path = lf_dir / _CONFIG_FILE
        gen_log_path = lf_dir / _GENERATION_LOG
        index_path = lf_dir / _INDEX_JSONL

        if config_path.exists():
            log_entries = self._index.load_log_entries(gen_log_path)
            incomplete = [e for e in log_entries if e.status == "pending"]
            if incomplete:
                logger.debug("RESUME: %d incomplete entries", len(incomplete))
                return ProjectMode.RESUME

        if index_path.exists() and self._fs.has_code_files(root):
            logger.debug("RESUME: existing index found")
            return ProjectMode.RESUME

        if not lf_dir.exists() and self._fs.has_code_files(root):
            logger.debug("EXPLAIN: code files found")
            return ProjectMode.EXPLAIN

        if self._fs.has_code_files(root) and not (lf_dir / _INDEX_JSONL).exists():
            logger.debug("EXPLAIN: index not yet built")
            return ProjectMode.EXPLAIN

        logger.debug("GENERATE: default")
        return ProjectMode.GENERATE

    def _load_or_create_config(self, root: Path, mode: ProjectMode) -> ProjectConfig:
        config_path = root / _LOCALFORGE_DIR / _CONFIG_FILE
        if config_path.exists():
            try:
                return ProjectConfig.model_validate_json(config_path.read_text(encoding="utf-8"))
            except (ValueError, json.JSONDecodeError) as exc:
                logger.warning("config.json parse error (using default): %s", exc)
        config = ProjectConfig(project_name=root.name, mode=mode, created_at=datetime.utcnow(), updated_at=datetime.utcnow())
        self.save_config(root, config)
        return config

    def save_config(self, root: Path, config: ProjectConfig) -> None:
        config_path = root / _LOCALFORGE_DIR / _CONFIG_FILE
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config.updated_at = datetime.utcnow()
        config_path.write_text(config.model_dump_json(indent=2), encoding="utf-8")

    def get_context_md(self, root: Path) -> str:
        context_path = root / _LOCALFORGE_DIR / _CONTEXT_FILE
        if context_path.exists():
            return context_path.read_text(encoding="utf-8")
        return ""

    def save_context_md(self, root: Path, content: str) -> None:
        context_path = root / _LOCALFORGE_DIR / _CONTEXT_FILE
        context_path.parent.mkdir(parents=True, exist_ok=True)
        context_path.write_text(content, encoding="utf-8")

    def save_generation_plan(self, root: Path, plan: GenerationPlan) -> None:
        plan_path = root / _LOCALFORGE_DIR / "plan.json"
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        plan_path.write_text(plan.model_dump_json(indent=2), encoding="utf-8")

    def load_generation_plan(self, root: Path) -> Optional[GenerationPlan]:
        plan_path = root / _LOCALFORGE_DIR / "plan.json"
        if not plan_path.exists():
            return None
        try:
            return GenerationPlan.model_validate_json(plan_path.read_text(encoding="utf-8"))
        except (ValueError, json.JSONDecodeError) as exc:
            logger.error("Plan parse error: %s", exc)
            return None

    def log_operation(self, root: Path, entry: GenerationLogEntry) -> None:
        self._index.append_log_entry(root / _LOCALFORGE_DIR / _GENERATION_LOG, entry)

    def update_log_entry_status(self, root: Path, file_path: str, status: str) -> None:
        log_path = root / _LOCALFORGE_DIR / _GENERATION_LOG
        entries = self._index.load_log_entries(log_path)
        updated = []
        for e in entries:
            if e.file_path == file_path and e.status == "pending":
                e.status = status
            updated.append(e)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as fh:
            for e in updated:
                fh.write(e.model_dump_json() + "\n")

    def get_file_tree(self, root: Path) -> List[FileNode]:
        return self._fs.build_file_tree(root)

    def get_project_status(self) -> Dict:
        if not self._current_project:
            return {"mode": None, "root": None, "model": None, "git_branch": None}
        p = self._current_project
        branch = self._git.get_current_branch(p.root)
        return {"mode": p.mode.value, "root": str(p.root), "model": p.config.model, "git_branch": branch or None}

    def _build_resume_state(self, root: Path) -> ResumeState:
        lf_dir = root / _LOCALFORGE_DIR
        is_localforge = (lf_dir / _CONFIG_FILE).exists()
        plan = self.load_generation_plan(root)
        completed: List[str] = []
        pending: List[str] = []
        if plan:
            log_entries = self._index.load_log_entries(lf_dir / _GENERATION_LOG)
            completed_paths = {e.file_path for e in log_entries if e.status == "completed"}
            for pf in plan.files:
                (completed if pf.path in completed_paths else pending).append(pf.path)
        else:
            code_files = self._fs.list_code_files(root)
            completed = [str(f.relative_to(root)) for f in code_files]
        git_log = self._git.get_log(root, max_entries=1)
        last_commit = git_log[0]["message"] if git_log else None
        return ResumeState(
            project_root=str(root), mode=ProjectMode.RESUME, plan=plan,
            completed_files=completed, pending_files=pending,
            last_commit_message=last_commit, is_localforge_project=is_localforge,
        )

    def set_model(self, root: Path, model: str) -> None:
        if self._current_project:
            self._current_project.config.model = model
            self.save_config(root, self._current_project.config)
