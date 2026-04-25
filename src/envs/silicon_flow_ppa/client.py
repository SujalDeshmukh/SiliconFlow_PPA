"""
SiliconFlow-PPA: HTTP Client.

This is what your RL training loop imports.
It communicates with the FastAPI server over HTTP.

Usage:
    env = SiliconFlowPPAClient(base_url="http://localhost:8000")
    result = env.reset()
    result = env.step(PPAAction(block_id="cpu", x=0.1, y=0.1, rotation=0))
"""
from __future__ import annotations
from typing import Optional

from core.http_env_client import HTTPEnvClient, StepResult
from envs.silicon_flow_ppa.models import (
    PPAAction, PPAObservation, PPAState,
)


class SiliconFlowPPAClient(HTTPEnvClient[PPAAction, PPAObservation]):
    """
    Type-safe HTTP client for the SiliconFlow-PPA environment server.
    """

    def __init__(self, base_url: str = "http://localhost:8000", timeout: int = 30):
        super().__init__(base_url=base_url, timeout=timeout)

    # ------------------------------------------------------------------
    # Required hooks
    # ------------------------------------------------------------------

    def _step_payload(self, action: PPAAction) -> dict:
        return action.to_dict()

    def _parse_result(self, payload: dict) -> StepResult[PPAObservation]:
        obs = PPAObservation.from_dict(payload)
        return StepResult(
            observation=obs,
            reward=obs.reward,
            done=obs.done,
            metadata=obs.metadata,
        )

    def _parse_state(self, payload: dict) -> PPAState:
        return PPAState.from_dict(payload)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def render(self) -> str:
        """Fetch and return the ASCII art render of the current layout."""
        import requests
        resp = requests.get(f"{self.base_url}/render", timeout=self.timeout)
        resp.raise_for_status()
        return resp.text

    def scenario_info(self) -> dict:
        import requests
        resp = requests.get(f"{self.base_url}/scenario", timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()