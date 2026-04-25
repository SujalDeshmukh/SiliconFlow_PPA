"""
SiliconFlow-PPA: Type-safe data contracts.

These dataclasses define the exact "shape" of information flowing
between the LLM agent and the chip-placement environment.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from core.env_server import Action, Observation, State


# ---------------------------------------------------------------------------
# Domain helpers
# ---------------------------------------------------------------------------

@dataclass
class LogicBlock:
    """A single hardware block to be placed on the chip die."""
    block_id: str           # e.g. "cpu_core_0"
    width: float            # normalised die-fraction width   (0-1)
    height: float           # normalised die-fraction height  (0-1)
    power_rating: float     # watts (used for thermal model)
    label: str = ""         # human-readable description


@dataclass
class NetConnection:
    """A single wire between two blocks (for wirelength calculation)."""
    source_id: str
    target_id: str
    weight: float = 1.0     # signal frequency / criticality


@dataclass
class PlacedBlock:
    """A block after the agent has placed it."""
    block_id: str
    x: float                # top-left corner, normalised (0-1)
    y: float
    width: float
    height: float
    power_rating: float
    rotation: int = 0       # 0 or 90 degrees


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

@dataclass
class PPAAction(Action):
    """
    The agent's single placement decision.

    The LLM outputs a JSON action; the client deserialises it here.
    """
    block_id: str = ""          # which block to place
    x: float = 0.0             # normalised x position (0-1)
    y: float = 0.0             # normalised y position (0-1)
    rotation: int = 0           # 0 = no rotation, 1 = 90°
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "block_id": self.block_id,
            "x": self.x,
            "y": self.y,
            "rotation": self.rotation,
            "metadata": self.metadata,
        }

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

@dataclass
class PPAObservation(Observation):
    """
    The "world state" the agent sees after each step.

    Contains:
      - die dimensions
      - remaining (unplaced) blocks
      - already placed blocks
      - netlist
      - incremental reward breakdown
    """
    # Die / canvas
    die_width: float = 1.0
    die_height: float = 1.0

    # Blocks
    remaining_blocks: List[LogicBlock] = field(default_factory=list)
    placed_blocks: List[PlacedBlock] = field(default_factory=list)

    # Connectivity
    netlist: List[NetConnection] = field(default_factory=list)

    # Step-level reward components (populated after each step)
    step_reward: float = 0.0
    is_legal: bool = True           # False if latest placement caused overlap / OOB
    overlap_penalty: float = 0.0

    # Running totals (for informational purposes)
    current_wirelength: float = 0.0
    current_area_util: float = 0.0  # fraction of die used
    current_thermal_max: float = 0.0

    # Standard OpenEnv fields
    done: bool = False
    reward: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "die_width": self.die_width,
            "die_height": self.die_height,
            "remaining_blocks": [b.__dict__ for b in self.remaining_blocks],
            "placed_blocks": [b.__dict__ for b in self.placed_blocks],
            "netlist": [n.__dict__ for n in self.netlist],
            "step_reward": self.step_reward,
            "is_legal": self.is_legal,
            "overlap_penalty": self.overlap_penalty,
            "current_wirelength": self.current_wirelength,
            "current_area_util": self.current_area_util,
            "current_thermal_max": self.current_thermal_max,
            "done": self.done,
            "reward": self.reward,
            "metadata": self.metadata,
        }

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

@dataclass
class PPAState(State):
    """Episode-level metadata."""
    episode_id: Optional[str] = None
    step_count: int = 0
    max_steps: int = 50
    total_blocks: int = 0
    placed_count: int = 0
    cumulative_reward: float = 0.0
    final_score: Optional[Dict[str, float]] = None  # set at episode end

    def to_dict(self) -> dict:
        return self.__dict__

    @classmethod
    def from_dict(cls, d: dict) -> "PPAState":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})