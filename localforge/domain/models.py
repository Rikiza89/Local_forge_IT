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


class ProjectMode(str, enum.Enum):
    GENERATE = "generate"
    RESUME = "resume"
    EXPLAIN = "explain"


class FileStatus(str, enum.Enum):
    PENDING = "pending"
    GENERATED = "generated"
    MODIFIED = "modified"
    INDEXED = "indexed"
    UNINDEXED = "unindexed"


class ChunkStrategy(str, enum.Enum):
    FULL = "full"
    HYBRID = "hybrid"


class FileNode(BaseModel):
    name: str
    path: str
    is_dir: bool
    status: FileStatus = FileStatus.UNINDEXED
    children: List["FileNode"] = Field(default_factory=list)
    size: Optional[int] = None
    modified_at: Optional[float] = None
    model_config = {"arbitrary_types_allowed": True}


class PlannedFile(BaseModel):
    path: str
    description: str
    dependencies: List[str] = Field(default_factory=list)


class GenerationPlan(BaseModel):
    project_name: str
    description: str
    files: List[PlannedFile]
    created_at: datetime = Field(default_factory=datetime.utcnow)
    approved: bool = False


class FileChunk(BaseModel):
    path: str
    content: str
    strategy: ChunkStrategy
    size: int
    mtime: float
    summary: Optional[str] = None
    language: Optional[str] = None
    indexed_at: Optional[datetime] = None


class ProjectIndex(BaseModel):
    project_root: str
    project_name: str
    summary: str
    file_chunks: List[FileChunk] = Field(default_factory=list)
    total_files: int = 0
    indexed_files: int = 0
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class ReportSection(BaseModel):
    name: str
    content: str


class ExplanationReport(BaseModel):
    project_root: str
    sections: List[ReportSection] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    complete: bool = False


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ResumeState(BaseModel):
    project_root: str
    mode: ProjectMode
    plan: Optional[GenerationPlan] = None
    completed_files: List[str] = Field(default_factory=list)
    pending_files: List[str] = Field(default_factory=list)
    last_commit_message: Optional[str] = None
    is_localforge_project: bool = False


class ProjectConfig(BaseModel):
    project_name: str = ""
    mode: ProjectMode = ProjectMode.GENERATE
    model: str = "llama3.2"
    token_limit: int = 6000
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class GenerationLogEntry(BaseModel):
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    mode: str
    model: str
    operation: str
    prompt_tokens_estimated: int = 0
    response_time_ms: Optional[float] = None
    file_path: Optional[str] = None
    status: str = "pending"


class Project(BaseModel):
    root: Path
    mode: ProjectMode
    config: ProjectConfig = Field(default_factory=ProjectConfig)
    file_tree: List[FileNode] = Field(default_factory=list)
    resume_state: Optional[ResumeState] = None
    index: Optional[ProjectIndex] = None
    model_config = {"arbitrary_types_allowed": True}
