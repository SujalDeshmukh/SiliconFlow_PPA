"""
SiliconFlow-PPA: ASCII Art Renderer.

Produces a terminal-friendly visualisation of the 2D chip layout.
"""
from __future__ import annotations
from typing import List

from envs.silicon_flow_ppa.models import LogicBlock, PlacedBlock

GRID_W = 40   # columns
GRID_H = 20   # rows
BLOCK_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


def render_ascii(
    placed: List[PlacedBlock],
    remaining: List[LogicBlock],
    die_w: float = 1.0,
    die_h: float = 1.0,
) -> str:
    grid = [["·"] * GRID_W for _ in range(GRID_H)]

    char_map = {}
    for i, b in enumerate(placed):
        ch = BLOCK_CHARS[i % len(BLOCK_CHARS)]
        char_map[b.block_id] = ch

        col_start = int(b.x / die_w * GRID_W)
        col_end   = min(GRID_W, int((b.x + b.width) / die_w * GRID_W))
        row_start = int(b.y / die_h * GRID_H)
        row_end   = min(GRID_H, int((b.y + b.height) / die_h * GRID_H))

        for row in range(row_start, row_end):
            for col in range(col_start, col_end):
                grid[row][col] = ch

    # Border
    border_top    = "┌" + "─" * GRID_W + "┐"
    border_bottom = "└" + "─" * GRID_W + "┘"
    lines = [border_top]
    for row in grid:
        lines.append("│" + "".join(row) + "│")
    lines.append(border_bottom)

    # Legend
    lines.append("")
    lines.append("Legend:")
    for b in placed:
        ch = char_map.get(b.block_id, "?")
        lines.append(f"  {ch} = {b.block_id:12s}  pos=({b.x:.2f},{b.y:.2f})  size={b.width:.2f}x{b.height:.2f}  pwr={b.power_rating:.1f}W")

    if remaining:
        lines.append("")
        lines.append("Unplaced blocks:")
        for b in remaining:
            lines.append(f"  - {b.block_id:12s}  size={b.width:.2f}x{b.height:.2f}  pwr={b.power_rating:.1f}W")

    return "\n".join(lines)