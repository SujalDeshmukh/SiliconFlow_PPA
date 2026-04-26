"""
OpenEnv Core: Generic HTTP client that communicates with an Environment server.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
import os
from typing import Any, Dict, Generic, Optional, TypeVar

import requests

ActionT = TypeVar("ActionT")
ObsT = TypeVar("ObsT")


@dataclass
class StepResult(Generic[ObsT]):
    observation: ObsT
    reward: Optional[float]
    done: bool
    metadata: Dict[str, Any] = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class HTTPEnvClient(ABC, Generic[ActionT, ObsT]):
    """
    Base class for all OpenEnv HTTP clients.

    Subclasses only need to implement:
      _step_payload(action) -> dict
      _parse_result(payload) -> StepResult
      _parse_state(payload)  -> State
    """

    def __init__(self, base_url: str = "http://localhost:8000", timeout: int = 30):
        resolved_url = os.environ.get("PPA_SERVER_URL", base_url)
        self.base_url = resolved_url.rstrip("/")
        self.timeout = timeout

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> StepResult:
        resp = requests.post(f"{self.base_url}/reset", timeout=self.timeout)
        resp.raise_for_status()
        return self._parse_result(resp.json())

    def step(self, action: ActionT) -> StepResult:
        payload = self._step_payload(action)
        resp = requests.post(
            f"{self.base_url}/step",
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return self._parse_result(resp.json())

    def state(self):
        resp = requests.get(f"{self.base_url}/state", timeout=self.timeout)
        resp.raise_for_status()
        return self._parse_state(resp.json())

    def health(self) -> bool:
        try:
            resp = requests.get(f"{self.base_url}/health", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Hooks for subclasses
    # ------------------------------------------------------------------

    @abstractmethod
    def _step_payload(self, action: ActionT) -> dict:
        """Convert a typed action to a JSON-serialisable dict."""

    @abstractmethod
    def _parse_result(self, payload: dict) -> StepResult:
        """Parse the HTTP JSON response into a typed StepResult."""

    @abstractmethod
    def _parse_state(self, payload: dict):
        """Parse the /state JSON response into a typed State object."""