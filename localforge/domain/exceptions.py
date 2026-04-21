"""
Custom exception classes used throughout the LocalForge application.
"""


class LocalForgeError(Exception):
    """Base exception for all LocalForge errors."""


class OllamaConnectionError(LocalForgeError):
    """Raised when HTTP connection to Ollama server fails."""


class OllamaModelNotFoundError(LocalForgeError):
    """Raised when the specified Ollama model is not found."""


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
    """Raised when the resume state file is corrupt or inconsistent."""
