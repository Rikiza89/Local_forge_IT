"""
Index adapter — ProjectIndex JSONL persistence.
Implements the IndexPort interface.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional

from localforge.domain.models import FileChunk, GenerationLogEntry, ProjectIndex

logger = logging.getLogger(__name__)


class IndexAdapter:

    def save_chunks(self, path: Path, chunks: List[FileChunk]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            for chunk in chunks:
                fh.write(chunk.model_dump_json() + "\n")
        logger.debug("Chunks saved: %d → %s", len(chunks), path)

    def load_chunks(self, path: Path) -> List[FileChunk]:
        if not path.exists():
            return []
        chunks: List[FileChunk] = []
        with path.open("r", encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    chunks.append(FileChunk.model_validate_json(line))
                except (json.JSONDecodeError, ValueError) as exc:
                    logger.warning("Chunk parse error (line %d): %s", lineno, exc)
        logger.debug("Chunks loaded: %d ← %s", len(chunks), path)
        return chunks

    def save_index(self, path: Path, index: ProjectIndex) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(index.model_dump_json(indent=2), encoding="utf-8")
        logger.debug("ProjectIndex saved: %s", path)

    def load_index(self, path: Path) -> Optional[ProjectIndex]:
        if not path.exists():
            return None
        try:
            return ProjectIndex.model_validate_json(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, ValueError) as exc:
            logger.error("ProjectIndex parse error: %s", exc)
            return None

    def append_log_entry(self, path: Path, entry: GenerationLogEntry) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(entry.model_dump_json() + "\n")

    def load_log_entries(self, path: Path) -> List[GenerationLogEntry]:
        if not path.exists():
            return []
        entries: List[GenerationLogEntry] = []
        with path.open("r", encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(GenerationLogEntry.model_validate_json(line))
                except (json.JSONDecodeError, ValueError) as exc:
                    logger.warning("Log entry parse error (line %d): %s", lineno, exc)
        return entries
