"""
Resume service — handles project resumption and diff detection.
Supports both LocalForge projects and foreign (external) projects.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Generator, List

from localforge.application.context_service import ContextService
from localforge.application.explanation_service import ExplanationService
from localforge.application.generation_service import GenerationService
from localforge.domain.models import GenerationPlan, Message, ResumeState
from localforge.infrastructure.filesystem_adapter import FileSystemAdapter
from localforge.infrastructure.git_adapter import GitAdapter

logger = logging.getLogger(__name__)

_LOCALFORGE_DIR = ".localforge"


class ResumeService:

    def __init__(self, fs: FileSystemAdapter, git: GitAdapter, generation: GenerationService, explanation: ExplanationService, context: ContextService) -> None:
        self._fs = fs
        self._git = git
        self._generation = generation
        self._explanation = explanation
        self._context = context

    def stream_continue_generation(self, root: Path, resume_state: ResumeState, model: str, context_md: str) -> Generator[dict, None, None]:
        if not resume_state.plan:
            yield {"error": "No generation plan found. Cannot resume."}
            return

        plan = resume_state.plan
        completed_set = set(resume_state.completed_files)
        start_from = 0

        for idx, pf in enumerate(plan.files):
            if pf.path not in completed_set:
                start_from = idx
                break

        logger.info("Resuming generation: continuing from file %d/%d", start_from, len(plan.files))

        yield from self._generation.stream_all_files(
            root=root, plan=plan, model=model, context_md=context_md, start_from=start_from,
        )

    def stream_foreign_qa(self, root: Path, model: str, question: str, history: List[Message]) -> Generator[dict, None, None]:
        yield from self._explanation.stream_answer(root=root, model=model, question=question, history=history)

    def get_resume_state_dict(self, resume_state: ResumeState) -> dict:
        plan_files = []
        if resume_state.plan:
            completed_set = set(resume_state.completed_files)
            for pf in resume_state.plan.files:
                plan_files.append({
                    "path": pf.path,
                    "description": pf.description,
                    "status": "completed" if pf.path in completed_set else "pending",
                })
        return {
            "is_localforge_project": resume_state.is_localforge_project,
            "completed_files": resume_state.completed_files,
            "pending_files": resume_state.pending_files,
            "last_commit_message": resume_state.last_commit_message,
            "plan_files": plan_files,
            "total_files": len(resume_state.completed_files) + len(resume_state.pending_files),
        }
