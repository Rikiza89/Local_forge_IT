"""
Ollama HTTP API wrapper — SSE streaming client.
The only class implementing the LLMPort interface.
"""

from __future__ import annotations

import json
import logging
import subprocess
from typing import Generator, List, Optional

import requests

from localforge.domain.exceptions import OllamaConnectionError, OllamaModelNotFoundError

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "http://localhost:11434"
_CONNECT_TIMEOUT = 5
_READ_TIMEOUT = 120
_GENERATE_READ_TIMEOUT = 600


def _detect_cuda() -> bool:
    """Check for a CUDA-capable GPU via nvidia-smi. Returns False on any error."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, timeout=5,
        )
        return result.returncode == 0 and bool(result.stdout.strip())
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False


class OllamaClient:

    def __init__(self, base_url: str = _DEFAULT_BASE_URL) -> None:
        self._base_url = base_url.rstrip("/")
        self._session = requests.Session()
        self._session.headers.update({"Content-Type": "application/json"})
        self.cuda_available: bool = _detect_cuda()
        if self.cuda_available:
            logger.info("CUDA GPU detected: setting num_gpu=-1 on Ollama requests")
        else:
            logger.info("No CUDA GPU detected: running in CPU inference mode")
        self.num_thread: Optional[int] = None

    def set_num_thread(self, num_thread: Optional[int]) -> None:
        self.num_thread = num_thread
        if num_thread is not None:
            logger.info("CPU thread count set: %d", num_thread)
        else:
            logger.info("CPU thread count reset to default (auto)")

    def is_available(self) -> bool:
        try:
            resp = self._session.get(f"{self._base_url}/api/tags", timeout=(_CONNECT_TIMEOUT, _CONNECT_TIMEOUT))
            return resp.status_code == 200
        except requests.RequestException:
            return False

    def list_models(self) -> List[str]:
        try:
            resp = self._session.get(f"{self._base_url}/api/tags", timeout=(_CONNECT_TIMEOUT, _READ_TIMEOUT))
            resp.raise_for_status()
            data = resp.json()
            return [m["name"] for m in data.get("models", [])]
        except requests.ConnectionError as exc:
            raise OllamaConnectionError(f"Cannot connect to Ollama server: {exc}") from exc
        except requests.RequestException as exc:
            raise OllamaConnectionError(f"Failed to retrieve model list: {exc}") from exc

    def stream_completion(self, model: str, prompt: str, system: Optional[str] = None, read_timeout: int = _READ_TIMEOUT) -> Generator[str, None, None]:
        payload: dict = {"model": model, "prompt": prompt, "stream": True}
        if system:
            payload["system"] = system
        options: dict = {}
        if self.cuda_available:
            options["num_gpu"] = -1
        if self.num_thread is not None:
            options["num_thread"] = self.num_thread
        if options:
            payload["options"] = options
        logger.debug("Ollama streaming start: model=%s", model)
        try:
            with self._session.post(f"{self._base_url}/api/generate", json=payload, stream=True, timeout=(_CONNECT_TIMEOUT, read_timeout)) as resp:
                if resp.status_code == 404:
                    raise OllamaModelNotFoundError(f"Model '{model}' not found. Run `ollama pull {model}` to fetch it.")
                resp.raise_for_status()
                for raw_line in resp.iter_lines():
                    if not raw_line:
                        continue
                    try:
                        chunk_data = json.loads(raw_line)
                    except json.JSONDecodeError as exc:
                        logger.warning("JSON decode error: %s — line: %r", exc, raw_line)
                        continue
                    token = chunk_data.get("response", "")
                    if token:
                        yield token
                    if chunk_data.get("done"):
                        logger.debug("Ollama streaming complete")
                        break
        except requests.ConnectionError as exc:
            raise OllamaConnectionError(f"Cannot connect to Ollama server: {exc}") from exc
        except (OllamaModelNotFoundError, OllamaConnectionError):
            raise
        except requests.RequestException as exc:
            raise OllamaConnectionError(f"Ollama request failed: {exc}") from exc

    def generate_sync(self, model: str, prompt: str, system: Optional[str] = None) -> str:
        """Non-streaming completion for tests and internal use."""
        return "".join(self.stream_completion(model, prompt, system, read_timeout=_GENERATE_READ_TIMEOUT))
