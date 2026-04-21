"""
Domain model definitions for the LocalForge application.
Uses Pydantic for type safety and validation.
"""

from __future__ import annotations

import enum
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class ProjectMode(str, enum.Enum):
    """Enum for project operating modes."""
    GENERATE = "generate"
    RESUME = "resume"
    EXPLAIN = "explain"


class FileStatus(str, enum.Enum):
    """Enum for file generation state."""
    PENDING = "pending"
    GENERATED = "generated"
    MODIFIED = "modified"
    INDEXED = "indexed"
    UNINDEXED = "unindexed"


class ChunkStrategy(str, enum.Enum):
    """Enum for file reading strategy."""
    FULL = "full"
    HYBRID = "hybrid"


# ---------------------------------------------------------------------------
# File node (for file tree display)
# ---------------------------------------------------------------------------

class FileNode(BaseModel):
    """Represents a single node in the file tree."""
    name: str
    path: str
    is_dir: bool
    status: FileStatus = FileStatus.UNINDEXED
    children: List["FileNode"] = Field(default_factory=list)
    size: Optional[int] = None
    modified_at: Optional[float] = None

    model_config = {"arbitrary_types_allowed": True}


# ---------------------------------------------------------------------------
# Generation plan
# ---------------------------------------------------------------------------

class PlannedFile(BaseModel):
    """A single file entry in a generation plan."""
    path: str
    description: str
    dependencies: List[str] = Field(default_factory=list)


class GenerationPlan(BaseModel):
    """AI-generated project structure plan."""
    project_name: str
    description: str
    files: List[PlannedFile]
    created_at: datetime = Field(default_factory=datetime.utcnow)
    approved: bool = False


# ---------------------------------------------------------------------------
# File chunk (for indexing)
# ---------------------------------------------------------------------------

class FileChunk(BaseModel):
    """File content chunk used during index building."""
    path: str
    content: str
    strategy: ChunkStrategy
    size: int
    mtime: float
    summary: Optional[str] = None
    language: Optional[str] = None
    indexed_at: Optional[datetime] = None


# ---------------------------------------------------------------------------
# ProjectIndex (master document)
# ---------------------------------------------------------------------------

class ProjectIndex(BaseModel):
    """Master document representing the full project index."""
    project_root: str
    project_name: str
    summary: str
    file_chunks: List[FileChunk] = Field(default_factory=list)
    total_files: int = 0
    indexed_files: int = 0
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Explanation report
# ---------------------------------------------------------------------------

class ReportSection(BaseModel):
    """A single section of an explanation report."""
    name: str
    content: str


class ExplanationReport(BaseModel):
    """Explanation report generated from codebase analysis."""
    project_root: str
    sections: List[ReportSection] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    complete: bool = False


# ---------------------------------------------------------------------------
# Chat history (Q&A)
# ---------------------------------------------------------------------------

class Message(BaseModel):
    """A single Q&A chat message."""
    role: Literal["user", "assistant"]
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Resume state
# ---------------------------------------------------------------------------

class ResumeState(BaseModel):
    """State snapshot used when resuming a project."""
    project_root: str
    mode: ProjectMode
    plan: Optional[GenerationPlan] = None
    completed_files: List[str] = Field(default_factory=list)
    pending_files: List[str] = Field(default_factory=list)
    last_commit_message: Optional[str] = None
    is_localforge_project: bool = False


# ---------------------------------------------------------------------------
# Project config (config.json)
# ---------------------------------------------------------------------------

class ProjectConfig(BaseModel):
    """Per-project settings stored in .localforge/config.json."""
    project_name: str = ""
    mode: ProjectMode = ProjectMode.GENERATE
    model: str = "llama3.2"
    token_limit: int = 6000
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Generation log entry
# ---------------------------------------------------------------------------

class GenerationLogEntry(BaseModel):
    """A single entry in generation_log.jsonl."""
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    mode: str
    model: str
    operation: str
    prompt_tokens_estimated: int = 0
    response_time_ms: Optional[float] = None
    file_path: Optional[str] = None
    status: str = "pending"


# ---------------------------------------------------------------------------
# Project (overall active state)
# ---------------------------------------------------------------------------

class Project(BaseModel):
    """Full state of the currently active project."""
    root: Path
    mode: ProjectMode
    config: ProjectConfig = Field(default_factory=ProjectConfig)
    file_tree: List[FileNode] = Field(default_factory=list)
    resume_state: Optional[ResumeState] = None
    index: Optional[ProjectIndex] = None

    model_config = {"arbitrary_types_allowed": True}
