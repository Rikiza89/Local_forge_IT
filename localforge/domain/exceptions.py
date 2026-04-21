"""
Custom exception classes used throughout the LocalForge application.
"""


class LocalForgeError(Exception):
    """Base exception for the LocalForge application."""


class OllamaConnectionError(LocalForgeError):
    """Raised when the HTTP connection to the Ollama server fails."""


class OllamaModelNotFoundError(LocalForgeError):
    """Raised when the specified Ollama model cannot be found."""


class PlanParseError(LocalForgeError):
    """Raised when JSON parsing of a generation plan fails."""


class FileWriteError(LocalForgeError):
    """Raised when writing a file fails."""


class GitOperationError(LocalForgeError):
    """Raised when a git operation fails."""


class ContextUpdateError(LocalForgeError):
    """Raised when updating context.md fails."""


class IndexBuildError(LocalForgeError):
    """Raised when building the ProjectIndex fails."""


class TokenBudgetExceededWarning(LocalForgeError):
    """
    Warning raised when the token budget is exceeded.
    Processing continues but the prompt is truncated.
    """


class ResumeStateCorruptError(LocalForgeError):
    """Raised when the resume state file is corrupted or inconsistent."""
