"""
SiliconFlow-PPA: Type-safe data contracts.

Uses Pydantic BaseModel for compatibility with openenv-core's
official create_fastapi_app which requires model_dump().
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

try:
    from openenv.core import Action, Observation, State
except ImportError:
    from core.env_server import Action, Observation, State


# ---------------------------------------------------------------------------
# Domain helpers (Pydantic models)
# ---------------------------------------------------------------------------

class LogicBlock(BaseModel):
    """A single hardware block to be placed on the chip die."""
    block_id: str
    width: float
    height: float
    power_rating: float
    label: str = ""


class NetConnection(BaseModel):
    """A single wire between two blocks (for wirelength calculation)."""
    source_id: str
    target_id: str
    weight: float = 1.0


class PlacedBlock(BaseModel):
    """A block after the agent has placed it."""
    block_id: str
    x: float
    y: float
    width: float
    height: float
    power_rating: float
    rotation: int = 0


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

class PPAAction(BaseModel):
    """The agent's single placement decision."""
    block_id: str = ""
    x: float = 0.0
    y: float = 0.0
    rotation: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> dict:
        return self.model_dump()

    @classmethod
    def from_dict(cls, d: dict) -> "PPAAction":
        return cls(
            block_id=d.get("block_id", ""),
            x=float(d.get("x", 0.0)),
            y=float(d.get("y", 0.0)),
            rotation=int(d.get("rotation", 0)),
            metadata=d.get("metadata", {}),
        )


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

class PPAObservation(BaseModel):
    """The world state the agent sees after each step."""

    # Die / canvas
    die_width: float = 1.0
    die_height: float = 1.0

    # Blocks
    remaining_blocks: List[LogicBlock] = Field(default_factory=list)
    placed_blocks: List[PlacedBlock] = Field(default_factory=list)

    # Connectivity
    netlist: List[NetConnection] = Field(default_factory=list)

    # Step-level reward components
    step_reward: float = 0.0
    is_legal: bool = True
    overlap_penalty: float = 0.0

    # Running totals
    current_wirelength: float = 0.0
    current_area_util: float = 0.0
    current_thermal_max: float = 0.0

    # Standard OpenEnv fields
    done: bool = False
    reward: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> dict:
        return self.model_dump()

    @classmethod
    def from_dict(cls, d: dict) -> "PPAObservation":
        return cls(
            die_width=d.get("die_width", 1.0),
            die_height=d.get("die_height", 1.0),
            remaining_blocks=[LogicBlock(**b) for b in d.get("remaining_blocks", [])],
            placed_blocks=[PlacedBlock(**b) for b in d.get("placed_blocks", [])],
            netlist=[NetConnection(**n) for n in d.get("netlist", [])],
            step_reward=d.get("step_reward", 0.0),
            is_legal=d.get("is_legal", True),
            overlap_penalty=d.get("overlap_penalty", 0.0),
            current_wirelength=d.get("current_wirelength", 0.0),
            current_area_util=d.get("current_area_util", 0.0),
            current_thermal_max=d.get("current_thermal_max", 0.0),
            done=d.get("done", False),
            reward=d.get("reward"),
            metadata=d.get("metadata", {}),
        )


# ---------------------------------------------------------------------------
# State (episode metadata)
# ---------------------------------------------------------------------------

class PPAState(BaseModel):
    """Episode-level metadata."""
    episode_id: Optional[str] = None
    step_count: int = 0
    max_steps: int = 50
    total_blocks: int = 0
    placed_count: int = 0
    cumulative_reward: float = 0.0
    final_score: Optional[Dict[str, float]] = None

    def to_dict(self) -> dict:
        return self.model_dump()

    @classmethod
    def from_dict(cls, d: dict) -> "PPAState":
        return cls(**{k: v for k, v in d.items() if k in cls.model_fields})