"""
SiliconFlow-PPA: Composable Reward Functions.

Each function is independent and returns a float.
The main reward is a weighted sum of all components.

Metrics:
  R1 - Functional Legality  (overlap / out-of-bounds check)
  R2 - Area Efficiency       (bounding-box utilisation)
  R3 - Wirelength            (total Manhattan distance of netlist)
  R4 - Thermal               (hotspot density)
  R5 - Format compliance     (did the LLM output valid JSON?)
  R6 - Completion bonus      (placed all blocks within step budget)
"""
from __future__ import annotations
import math
from typing import List

from envs.silicon_flow_ppa.models import (
    LogicBlock, NetConnection, PlacedBlock, PPAObservation,
)


# ---------------------------------------------------------------------------
# Weights (tunable)
# ---------------------------------------------------------------------------
WEIGHTS = {
    "legality":      2.0,   # heavy penalty for illegal placement
    "area":          1.0,
    "wirelength":    1.0,
    "thermal":       1.5,
    "completion":    3.0,   # big bonus for finishing the layout
}

THERMAL_THRESHOLD = 0.6    # max allowed normalised power density per cell
OVERLAP_PENALTY   = -5.0   # applied per illegal placement


# ---------------------------------------------------------------------------
# R1 – Functional Legality
# ---------------------------------------------------------------------------

def check_legality(
    new_block: PlacedBlock,
    existing_blocks: List[PlacedBlock],
    die_width: float = 1.0,
    die_height: float = 1.0,
) -> tuple[bool, float]:
    """
    Returns (is_legal, penalty).
    Penalty is 0.0 if legal, OVERLAP_PENALTY if not.
    """
    # Out-of-bounds check
    if (
        new_block.x < 0
        or new_block.y < 0
        or new_block.x + new_block.width  > die_width  + 1e-6
        or new_block.y + new_block.height > die_height + 1e-6
    ):
        return False, OVERLAP_PENALTY

    # Overlap with existing blocks (axis-aligned rectangle intersection)
    for b in existing_blocks:
        if _rects_overlap(new_block, b):
            return False, OVERLAP_PENALTY

    return True, 0.0


def _rects_overlap(a: PlacedBlock, b: PlacedBlock) -> bool:
    return not (
        a.x + a.width  <= b.x
        or b.x + b.width  <= a.x
        or a.y + a.height <= b.y
        or b.y + b.height <= a.y
    )


# ---------------------------------------------------------------------------
# R2 – Area Efficiency
# ---------------------------------------------------------------------------

def area_efficiency_reward(
    placed_blocks: List[PlacedBlock],
    die_width: float = 1.0,
    die_height: float = 1.0,
) -> float:
    """
    Reward inversely proportional to the bounding-box of all placed blocks
    relative to the die area.  Score in [0, 1].
    """
    if not placed_blocks:
        return 0.0

    min_x = min(b.x for b in placed_blocks)
    min_y = min(b.y for b in placed_blocks)
    max_x = max(b.x + b.width  for b in placed_blocks)
    max_y = max(b.y + b.height for b in placed_blocks)

    bbox_area    = (max_x - min_x) * (max_y - min_y)
    die_area     = die_width * die_height
    utilisation  = bbox_area / die_area if die_area > 0 else 1.0

    # 1.0 = perfectly compact, 0.0 = spread over whole die
    return max(0.0, 1.0 - utilisation)


# ---------------------------------------------------------------------------
# R3 – Wirelength (total Manhattan distance)
# ---------------------------------------------------------------------------

def wirelength_reward(
    placed_blocks: List[PlacedBlock],
    netlist: List[NetConnection],
) -> float:
    """
    Reward is high when total wirelength is low.
    Score in [0, 1] (normalised by worst-case diagonal distance * num_wires).
    """
    if not netlist or not placed_blocks:
        return 0.0

    centers = {
        b.block_id: (b.x + b.width / 2, b.y + b.height / 2)
        for b in placed_blocks
    }

    total_wl   = 0.0
    worst_case = 0.0

    for conn in netlist:
        src = centers.get(conn.source_id)
        dst = centers.get(conn.target_id)
        if src and dst:
            dist       = abs(src[0] - dst[0]) + abs(src[1] - dst[1])
            total_wl  += dist * conn.weight
            worst_case += math.sqrt(2) * conn.weight   # max diagonal

    if worst_case == 0:
        return 0.0

    return max(0.0, 1.0 - total_wl / worst_case)


def compute_total_wirelength(
    placed_blocks: List[PlacedBlock],
    netlist: List[NetConnection],
) -> float:
    """Raw total wirelength (for logging)."""
    centers = {
        b.block_id: (b.x + b.width / 2, b.y + b.height / 2)
        for b in placed_blocks
    }
    total = 0.0
    for conn in netlist:
        src = centers.get(conn.source_id)
        dst = centers.get(conn.target_id)
        if src and dst:
            total += (abs(src[0] - dst[0]) + abs(src[1] - dst[1])) * conn.weight
    return total


# ---------------------------------------------------------------------------
# R4 – Thermal Density
# ---------------------------------------------------------------------------

def thermal_reward(
    placed_blocks: List[PlacedBlock],
    die_width: float = 1.0,
    die_height: float = 1.0,
    grid_resolution: int = 10,
) -> tuple[float, float]:
    """
    Discretise die into grid_resolution × grid_resolution cells.
    Compute normalised power density per cell.
    Return (reward, max_density).
    Reward = 1 if max_density <= threshold, falls off above it.
    """
    if not placed_blocks:
        return 1.0, 0.0

    grid = [[0.0] * grid_resolution for _ in range(grid_resolution)]
    cell_w = die_width  / grid_resolution
    cell_h = die_height / grid_resolution

    for b in placed_blocks:
        # find which grid cells the block overlaps
        col_start = max(0, int(b.x / cell_w))
        col_end   = min(grid_resolution - 1, int((b.x + b.width)  / cell_w))
        row_start = max(0, int(b.y / cell_h))
        row_end   = min(grid_resolution - 1, int((b.y + b.height) / cell_h))

        block_area = b.width * b.height
        if block_area <= 0:
            continue

        for row in range(row_start, row_end + 1):
            for col in range(col_start, col_end + 1):
                # fraction of block in this cell
                cx0 = col * cell_w;   cx1 = cx0 + cell_w
                cy0 = row * cell_h;   cy1 = cy0 + cell_h
                overlap_w = max(0, min(b.x + b.width,  cx1) - max(b.x, cx0))
                overlap_h = max(0, min(b.y + b.height, cy1) - max(b.y, cy0))
                fraction  = (overlap_w * overlap_h) / block_area
                grid[row][col] += b.power_rating * fraction

    max_density = max(grid[r][c] for r in range(grid_resolution) for c in range(grid_resolution))

    if max_density <= THERMAL_THRESHOLD:
        reward = 1.0
    else:
        reward = max(0.0, 1.0 - (max_density - THERMAL_THRESHOLD) / THERMAL_THRESHOLD)

    return reward, max_density


# ---------------------------------------------------------------------------
# Combined Reward
# ---------------------------------------------------------------------------

def compute_step_reward(
    obs: PPAObservation,
    is_final: bool = False,
) -> float:
    """
    Weighted combination of all sub-rewards.
    Called once per step; at episode end is_final=True applies completion bonus.
    """
    w = WEIGHTS

    # R1
    legality_r = 0.0 if obs.is_legal else 1.0
    legality_r = (1.0 - legality_r)  # 1 if legal, 0 if not

    # R2
    area_r = area_efficiency_reward(obs.placed_blocks, obs.die_width, obs.die_height)

    # R3
    wire_r = wirelength_reward(obs.placed_blocks, obs.netlist)

    # R4
    therm_r, _ = thermal_reward(obs.placed_blocks, obs.die_width, obs.die_height)

    # Penalty from illegal placement
    penalty = obs.overlap_penalty

    score = (
        w["legality"]   * legality_r
        + w["area"]       * area_r
        + w["wirelength"] * wire_r
        + w["thermal"]    * therm_r
        + penalty
    )

    if is_final and obs.is_legal:
        score += w["completion"]

    return round(score, 4)