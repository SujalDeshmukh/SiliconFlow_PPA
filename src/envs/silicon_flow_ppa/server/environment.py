"""
SiliconFlow-PPA: Core Environment Implementation.

Simulates a 2D chip die where an LLM agent places logic blocks
one at a time, receiving PPA (Power, Performance, Area) feedback.
"""
from __future__ import annotations
import copy
import os
import random
import uuid
from typing import Any, Dict, List, Optional

from openenv.core import Environment, Action

from envs.silicon_flow_ppa.models import (
    LogicBlock, NetConnection, PlacedBlock,
    PPAAction, PPAObservation, PPAState,
)
from envs.silicon_flow_ppa.rewards import (
    area_efficiency_reward,
    check_legality,
    compute_step_reward,
    congestion_reward,
    timing_reward,
    compute_total_wirelength,
    thermal_reward,
)
from envs.silicon_flow_ppa.config import load_json


# ---------------------------------------------------------------------------
# Scenario catalogue — each scenario is a dict with blocks + netlist
# ---------------------------------------------------------------------------

DEFAULT_SCENARIOS = {
    "small_4block": {
        "blocks": [
            {"block_id": "cpu",    "width": 0.25, "height": 0.25, "power_rating": 5.0, "label": "CPU Core"},
            {"block_id": "gpu",    "width": 0.30, "height": 0.30, "power_rating": 8.0, "label": "GPU Core"},
            {"block_id": "mem",    "width": 0.20, "height": 0.15, "power_rating": 2.0, "label": "Memory Controller"},
            {"block_id": "io",     "width": 0.15, "height": 0.15, "power_rating": 1.0, "label": "I/O Block"},
        ],
        "netlist": [
            {"source_id": "cpu", "target_id": "mem",  "weight": 2.0},
            {"source_id": "cpu", "target_id": "gpu",  "weight": 1.5},
            {"source_id": "gpu", "target_id": "mem",  "weight": 1.0},
            {"source_id": "cpu", "target_id": "io",   "weight": 0.5},
        ],
    },
    "medium_6block": {
        "blocks": [
            {"block_id": "cpu0",   "width": 0.20, "height": 0.20, "power_rating": 4.0, "label": "CPU Core 0"},
            {"block_id": "cpu1",   "width": 0.20, "height": 0.20, "power_rating": 4.0, "label": "CPU Core 1"},
            {"block_id": "gpu",    "width": 0.25, "height": 0.25, "power_rating": 7.0, "label": "GPU"},
            {"block_id": "cache",  "width": 0.15, "height": 0.10, "power_rating": 1.5, "label": "L3 Cache"},
            {"block_id": "pcie",   "width": 0.10, "height": 0.10, "power_rating": 0.5, "label": "PCIe Controller"},
            {"block_id": "ddr",    "width": 0.20, "height": 0.10, "power_rating": 2.0, "label": "DDR Interface"},
        ],
        "netlist": [
            {"source_id": "cpu0",  "target_id": "cache",  "weight": 3.0},
            {"source_id": "cpu1",  "target_id": "cache",  "weight": 3.0},
            {"source_id": "cache", "target_id": "ddr",    "weight": 2.0},
            {"source_id": "cpu0",  "target_id": "gpu",    "weight": 1.0},
            {"source_id": "gpu",   "target_id": "pcie",   "weight": 1.5},
        ],
    },
    "hard_8block": {
        "blocks": [
            {"block_id": "a53_0",  "width": 0.15, "height": 0.15, "power_rating": 2.0, "label": "A53 Core 0"},
            {"block_id": "a53_1",  "width": 0.15, "height": 0.15, "power_rating": 2.0, "label": "A53 Core 1"},
            {"block_id": "a78",    "width": 0.18, "height": 0.18, "power_rating": 5.0, "label": "A78 Big Core"},
            {"block_id": "mali",   "width": 0.20, "height": 0.20, "power_rating": 6.0, "label": "Mali GPU"},
            {"block_id": "npu",    "width": 0.12, "height": 0.12, "power_rating": 3.0, "label": "Neural Engine"},
            {"block_id": "isp",    "width": 0.10, "height": 0.10, "power_rating": 1.5, "label": "Image Signal Proc"},
            {"block_id": "modem",  "width": 0.15, "height": 0.12, "power_rating": 4.0, "label": "5G Modem"},
            {"block_id": "pmu",    "width": 0.10, "height": 0.08, "power_rating": 0.5, "label": "Power Mgmt Unit"},
        ],
        "netlist": [
            {"source_id": "a53_0", "target_id": "a53_1",  "weight": 1.0},
            {"source_id": "a53_0", "target_id": "a78",    "weight": 1.5},
            {"source_id": "a78",   "target_id": "mali",   "weight": 2.0},
            {"source_id": "a78",   "target_id": "npu",    "weight": 2.5},
            {"source_id": "mali",  "target_id": "isp",    "weight": 1.0},
            {"source_id": "modem", "target_id": "a53_0",  "weight": 1.0},
            {"source_id": "pmu",   "target_id": "a78",    "weight": 0.5},
            {"source_id": "npu",   "target_id": "isp",    "weight": 1.5},
        ],
    },
}


def _load_scenarios() -> Dict[str, Dict[str, Any]]:
    configured_path = os.environ.get("PPA_SCENARIOS_FILE")
    try:
        payload = load_json(configured_path, "configs/scenarios.json")
        return payload if payload else DEFAULT_SCENARIOS
    except Exception:
        return DEFAULT_SCENARIOS


SCENARIOS = _load_scenarios()


class SiliconFlowPPAEnvironment(Environment):
    """
    2-D chip placement environment for RL training.

    Episode flow:
      reset() → returns initial observation with all blocks unplaced
      step()  → agent places one block; returns updated observation + reward
      ...     → repeats until all blocks placed OR max_steps reached
    """

    def __init__(
        self,
        scenario: str = "small_4block",
        max_steps: int = 50,
        die_width: float = 1.0,
        die_height: float = 1.0,
        randomise_scenario: bool = False,
    ):
        self.scenario_name = scenario
        self.max_steps = max_steps
        self.die_width = die_width
        self.die_height = die_height
        self.randomise_scenario = randomise_scenario

        self._reset_internals()

    # ------------------------------------------------------------------
    # OpenEnv required interface
    # ------------------------------------------------------------------

    def reset(self) -> PPAObservation:
        if self.randomise_scenario:
            self.scenario_name = random.choice(list(SCENARIOS.keys()))

        self._reset_internals()
        self._episode_id = str(uuid.uuid4())[:8]

        return self._make_observation(
            step_reward=0.0,
            is_legal=True,
            overlap_penalty=0.0,
        )

    def step(self, action: PPAAction) -> PPAObservation:  # type: ignore[override]
        if self._done:
            # Episode already finished; return terminal state
            return self._make_observation(step_reward=0.0, is_legal=True, overlap_penalty=0.0)

        self._step_count += 1

        # ---- 1. Resolve the block being placed ----
        target = next(
            (b for b in self._remaining_blocks if b.block_id == action.block_id),
            None,
        )
        if target is None:
            # Unknown / already-placed block – penalise format error
            obs = self._make_observation(step_reward=-1.0, is_legal=False, overlap_penalty=-1.0)
            obs.metadata["error"] = f"block_id '{action.block_id}' not found in remaining blocks"
            return obs

        # ---- 2. Apply rotation ----
        w, h = target.width, target.height
        if action.rotation == 1:
            w, h = h, w

        candidate = PlacedBlock(
            block_id=target.block_id,
            x=action.x,
            y=action.y,
            width=w,
            height=h,
            power_rating=target.power_rating,
            rotation=action.rotation,
        )

        # ---- 3. Legality check ----
        is_legal, penalty = check_legality(
            candidate, self._placed_blocks, self.die_width, self.die_height
        )

        if is_legal:
            self._placed_blocks.append(candidate)
            self._remaining_blocks = [
                b for b in self._remaining_blocks if b.block_id != action.block_id
            ]

        # ---- 4. Check termination ----
        is_final = len(self._remaining_blocks) == 0 or self._step_count >= self.max_steps
        self._done = is_final

        # ---- 5. Build observation and compute reward ----
        obs = self._make_observation(
            step_reward=0.0,
            is_legal=is_legal,
            overlap_penalty=penalty,
        )
        obs.reward = compute_step_reward(obs, is_final=is_final)
        obs.step_reward = obs.reward
        obs.done = is_final
        self._cumulative_reward += obs.reward

        return obs

    def _parse_action(self, payload: dict) -> PPAAction:
        """Called by the FastAPI server to deserialise incoming JSON."""
        return PPAAction.from_dict(payload)

    @property
    def state(self) -> PPAState:
        return PPAState(
            episode_id=self._episode_id,
            step_count=self._step_count,
            max_steps=self.max_steps,
            total_blocks=self._total_blocks,
            placed_count=len(self._placed_blocks),
            cumulative_reward=self._cumulative_reward,
            final_score=self._final_score,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _reset_internals(self):
        scenario = SCENARIOS.get(self.scenario_name, SCENARIOS["small_4block"])
        self._remaining_blocks: List[LogicBlock] = [
            LogicBlock(**b) for b in scenario["blocks"]
        ]
        self._placed_blocks: List[PlacedBlock] = []
        self._netlist: List[NetConnection] = [
            NetConnection(**n) for n in scenario["netlist"]
        ]
        self._step_count = 0
        self._done = False
        self._episode_id = "init"
        self._total_blocks = len(self._remaining_blocks)
        self._cumulative_reward = 0.0
        self._final_score: Optional[Dict[str, float]] = None

    def _make_observation(
        self,
        step_reward: float,
        is_legal: bool,
        overlap_penalty: float,
    ) -> PPAObservation:
        # Compute running metrics
        wl = compute_total_wirelength(self._placed_blocks, self._netlist)
        area_util = area_efficiency_reward(self._placed_blocks, self.die_width, self.die_height)
        _, therm_max = thermal_reward(self._placed_blocks, self.die_width, self.die_height)
        _, congestion_max = congestion_reward(self._placed_blocks, self._netlist, self.die_width, self.die_height)
        _, timing_cost = timing_reward(self._placed_blocks, self._netlist, self.die_width, self.die_height)

        obs = PPAObservation(
            die_width=self.die_width,
            die_height=self.die_height,
            remaining_blocks=copy.deepcopy(self._remaining_blocks),
            placed_blocks=copy.deepcopy(self._placed_blocks),
            netlist=copy.deepcopy(self._netlist),
            step_reward=step_reward,
            is_legal=is_legal,
            overlap_penalty=overlap_penalty,
            current_wirelength=wl,
            current_area_util=area_util,
            current_thermal_max=therm_max,
            current_congestion=congestion_max,
            current_timing_cost=timing_cost,
            done=self._done,
            reward=step_reward,
        )
        return obs