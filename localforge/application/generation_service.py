"""
Generation service — orchestrates file generation from a plan.
Generates, writes, and commits files sequentially.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Generator, List, Optional

from localforge.application.context_service import ContextService
from localforge.domain.exceptions import PlanParseError
from localforge.domain.models import GenerationLogEntry, GenerationPlan, PlannedFile
from localforge.infrastructure.filesystem_adapter import FileSystemAdapter
from localforge.infrastructure.git_adapter import GitAdapter
from localforge.infrastructure.index_adapter import IndexAdapter
from localforge.infrastructure.ollama_client import OllamaClient

logger = logging.getLogger(__name__)

_LOCALFORGE_DIR = ".localforge"
_cancel_flag: bool = False


def request_cancel() -> None:
    global _cancel_flag
    _cancel_flag = True
    logger.info("Generation cancel requested")


def reset_cancel() -> None:
    global _cancel_flag
    _cancel_flag = False


class GenerationService:
    """Service responsible for AI-driven project file generation."""

    def __init__(self, fs: FileSystemAdapter, git: GitAdapter, index_adapter: IndexAdapter, llm: OllamaClient, context: ContextService) -> None:
        self._fs = fs
        self._git = git
        self._index_adapter = index_adapter
        self._llm = llm
        self._context = context

    def stream_plan(self, root, model, user_prompt, folder_name, file_tree_text, context_md, git_log) -> Generator[dict, None, None]:
        reset_cancel()
        prompt = self._context.build_plan_prompt(
            user_prompt=user_prompt, folder_name=folder_name,
            file_tree_text=file_tree_text, context_md=context_md, git_log=git_log,
        )
        start_time = time.time()
        try:
            for token in self._llm.stream_completion(model, prompt):
                if _cancel_flag:
                    yield {"error": "Generazione annullata"}
                    return
                yield {"token": token}
        except Exception as exc:
            logger.error("Plan generation error: %s", exc)
            yield {"error": str(exc)}
            return

        elapsed = (time.time() - start_time) * 1000
        log_entry = GenerationLogEntry(mode="generate", model=model, operation="plan", response_time_ms=elapsed)
        self._index_adapter.append_log_entry(root / _LOCALFORGE_DIR / "generation_log.jsonl", log_entry)
        yield {"done": True}

    def parse_plan(self, plan_text: str) -> GenerationPlan:
        text = plan_text.strip()
        if "```json" in text:
            start = text.index("```json") + 7
            end = text.index("```", start)
            text = text[start:end].strip()
        elif "```" in text:
            start = text.index("```") + 3
            end = text.index("```", start)
            text = text[start:end].strip()

        json_start = text.find("{")
        json_end = text.rfind("}") + 1
        if json_start >= 0 and json_end > json_start:
            text = text[json_start:json_end]

        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise PlanParseError(f"Failed to parse plan JSON: {exc}") from exc

        try:
            return GenerationPlan(
                project_name=data.get("project_name", "unnamed"),
                description=data.get("description", ""),
                files=[
                    PlannedFile(
                        path=f.get("path", ""),
                        description=f.get("description", ""),
                        dependencies=f.get("dependencies", []),
                    )
                    for f in data.get("files", [])
                ],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise PlanParseError(f"Invalid plan structure: {exc}") from exc

    def stream_all_files(self, root, plan, model, context_md, start_from=None) -> Generator[dict, None, None]:
        reset_cancel()
        files = plan.files
        total = len(files)
        plan_json = plan.model_dump_json(indent=2)
        start_idx = start_from or 0

        if not (root / ".git").is_dir():
            try:
                self._git.init(root)
            except Exception as exc:
                logger.warning("git init failed: %s", exc)

        for idx, planned_file in enumerate(files[start_idx:], start=start_idx):
            if _cancel_flag:
                yield {"error": "Generazione annullata"}
                return

            yield {"progress": {"done": idx, "total": total, "current_file": planned_file.path}}

            dependency_contents: List[tuple[str, str]] = []
            for dep_path in planned_file.dependencies:
                dep_full = root / dep_path
                if dep_full.exists():
                    try:
                        content = self._fs.read_text(dep_full)
                        dependency_contents.append((dep_path, content))
                    except Exception:
                        pass

            prompt = self._context.build_file_generation_prompt(
                target_file=planned_file.path, target_description=planned_file.description,
                context_md=context_md, plan_json=plan_json, dependency_contents=dependency_contents,
            )

            log_path = root / _LOCALFORGE_DIR / "generation_log.jsonl"
            log_entry = GenerationLogEntry(mode="generate", model=model, operation="generate_file", file_path=planned_file.path, status="pending")
            self._index_adapter.append_log_entry(log_path, log_entry)

            start_time = time.time()
            file_content_parts: List[str] = []

            try:
                for token in self._llm.stream_completion(model, prompt):
                    if _cancel_flag:
                        yield {"error": "Generazione annullata"}
                        return
                    file_content_parts.append(token)
                    yield {"token": token}
            except Exception as exc:
                logger.error("File generation error [%s]: %s", planned_file.path, exc)
                yield {"error": str(exc)}
                continue

            elapsed = (time.time() - start_time) * 1000
            file_content = "".join(file_content_parts)

            file_path = root / planned_file.path
            try:
                self._fs.write_text(file_path, file_content)
            except Exception as exc:
                logger.error("File write error [%s]: %s", planned_file.path, exc)
                yield {"error": str(exc)}
                continue

            try:
                self._git.commit_all(root, f"LocalForge: genera {planned_file.path}")
            except Exception as exc:
                logger.warning("git commit failed [%s]: %s", planned_file.path, exc)

            self._update_log_status(log_path, planned_file.path, elapsed)
            yield {"file_written": planned_file.path}

        yield {"progress": {"done": total, "total": total, "current_file": "completato"}}
        yield {"done": True}

    def stream_regenerate_file(self, root, plan, model, context_md, file_path) -> Generator[dict, None, None]:
        reset_cancel()
        planned_file = next((f for f in plan.files if f.path == file_path), None)
        if not planned_file:
            yield {"error": f"File non trovato nel piano: {file_path}"}
            return

        dependency_contents: List[tuple[str, str]] = []
        for dep_path in planned_file.dependencies:
            dep_full = root / dep_path
            if dep_full.exists():
                try:
                    content = self._fs.read_text(dep_full)
                    dependency_contents.append((dep_path, content))
                except Exception:
                    pass

        prompt = self._context.build_file_generation_prompt(
            target_file=planned_file.path, target_description=planned_file.description,
            context_md=context_md, plan_json=plan.model_dump_json(indent=2),
            dependency_contents=dependency_contents,
        )

        file_content_parts: List[str] = []
        try:
            for token in self._llm.stream_completion(model, prompt):
                if _cancel_flag:
                    yield {"error": "Generazione annullata"}
                    return
                file_content_parts.append(token)
                yield {"token": token}
        except Exception as exc:
            yield {"error": str(exc)}
            return

        file_content = "".join(file_content_parts)
        full_path = root / planned_file.path
        try:
            self._fs.write_text(full_path, file_content)
        except Exception as exc:
            yield {"error": str(exc)}
            return

        try:
            self._git.commit_all(root, f"LocalForge: rigenera {planned_file.path}")
        except Exception as exc:
            logger.warning("git commit failed [%s]: %s", planned_file.path, exc)

        yield {"file_written": planned_file.path}
        yield {"done": True}

    def _update_log_status(self, log_path: Path, file_path: str, elapsed_ms: float) -> None:
        if not log_path.exists():
            return
        entries = self._index_adapter.load_log_entries(log_path)
        updated = []
        last_updated = False
        for e in reversed(entries):
            if not last_updated and e.file_path == file_path and e.status == "pending":
                e.status = "completed"
                e.response_time_ms = elapsed_ms
                last_updated = True
            updated.append(e)
        updated.reverse()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as fh:
            for e in updated:
                fh.write(e.model_dump_json() + "\n")
