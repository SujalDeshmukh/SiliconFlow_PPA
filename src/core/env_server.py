"""
OpenEnv Core: Abstract base classes for Environment, Action, Observation, State.
Matches the OpenEnv spec from meta-pytorch/OpenEnv.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Generic, List, Optional, TypeVar

import fastapi
from fastapi import FastAPI
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Abstract data contracts
# ---------------------------------------------------------------------------

@dataclass
class Action(ABC):
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Observation(ABC):
    done: bool = False
    reward: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class State(ABC):
    episode_id: Optional[str] = None
    step_count: int = 0


# ---------------------------------------------------------------------------
# Abstract Environment
# ---------------------------------------------------------------------------

class Environment(ABC):
    """Base class for all OpenEnv environment implementations."""

    @abstractmethod
    def reset(self) -> Observation:
        """Start a new episode and return the initial observation."""

    @abstractmethod
    def step(self, action: Action) -> Observation:
        """Execute an action and return the resulting observation."""

    @property
    @abstractmethod
    def state(self) -> State:
        """Return current episode metadata (read-only)."""


# ---------------------------------------------------------------------------
# FastAPI app factory
# ---------------------------------------------------------------------------

def create_fastapi_app(env: Environment) -> FastAPI:
    """
    Wrap an Environment in a FastAPI app exposing:
      POST /reset
      POST /step
      GET  /state
    """
    app = FastAPI(title="OpenEnv Environment Server")

    @app.post("/reset")
    def reset():
        obs = env.reset()
        return obs.__dict__

    @app.post("/step")
    def step(payload: dict):
        # Delegate payload parsing to the concrete environment
        action = env._parse_action(payload)  # type: ignore[attr-defined]
        obs = env.step(action)
        return obs.__dict__

    @app.get("/state")
    def state():
        return env.state.__dict__

    @app.get("/health")
    def health():
        return {"status": "ok"}

    return app