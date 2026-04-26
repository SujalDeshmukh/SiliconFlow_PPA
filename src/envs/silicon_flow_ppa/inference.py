"""
SiliconFlow-PPA: LLM Inference Script.

This module wraps an LLM (via the Anthropic/OpenAI API or a local model)
to act as the placement agent. It receives a PPAObservation and outputs
a PPAAction by prompting the LLM to reason about the chip layout.

The LLM is expected to return a JSON action like:
  {"block_id": "cpu", "x": 0.1, "y": 0.1, "rotation": 0}
"""
from __future__ import annotations
import json
import os
import re
from typing import Optional

from envs.silicon_flow_ppa.models import (
    LogicBlock, PPAAction, PPAObservation,
)


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are SiliconFlow-PPA, an expert chip physical design engineer.

Place logic blocks on a 2D die (normalised 0-1 × 0-1) to minimise wirelength,
maximise compactness, and prevent thermal hotspots.

CRITICAL RULES (violations waste steps and reduce reward):
1. NEVER place a block where it overlaps an already placed block.
   - Check: new_x + new_width must not overlap any placed block's x range
   - Check: new_y + new_height must not overlap any placed block's y range
2. Block must fit within die: x + width <= 1.0, y + height <= 1.0
3. Place only blocks listed in "remaining_blocks".

STRATEGY — think step by step:
1. Pack blocks left-to-right along y=0 first for best area utilisation.
2. Next block x = max(placed_block.x + placed_block.width) to avoid overlap.
3. Space high-power blocks to reduce thermal hotspots.
4. Place connected blocks (high netlist weight) close together.

Respond with ONLY a JSON object — no explanation, no markdown:
{"block_id": "<id>", "x": <float>, "y": <float>, "rotation": <0 or 1>}
"""


def _build_user_prompt(obs: PPAObservation) -> str:
    """Serialise the current observation into a concise prompt for the LLM."""
    
    # Calculate next safe x position
    next_x = 0.0
    if obs.placed_blocks:
        next_x = max(round(b.x + b.width, 3) for b in obs.placed_blocks)
    
    # List occupied regions clearly
    occupied = [
        f"{b.block_id}: x=[{round(b.x,3)} to {round(b.x+b.width,3)}], y=[{round(b.y,3)} to {round(b.y+b.height,3)}]"
        for b in obs.placed_blocks
    ]

    data = {
        "die": {"width": obs.die_width, "height": obs.die_height},
        "placement_hint": {
            "next_safe_x": next_x,
            "advice": f"Place next block at x={next_x}, y=0.0 to avoid overlaps. Never reuse a failed position.",
            "occupied_regions": occupied,
        },
        "remaining_blocks": [
            {
                "block_id": b.block_id,
                "width": round(b.width, 3),
                "height": round(b.height, 3),
                "power_rating": b.power_rating,
                "label": b.label,
            }
            for b in obs.remaining_blocks
        ],
        "placed_blocks": [
            {
                "block_id": b.block_id,
                "x": round(b.x, 3),
                "y": round(b.y, 3),
                "width": round(b.width, 3),
                "height": round(b.height, 3),
                "power_rating": b.power_rating,
            }
            for b in obs.placed_blocks
        ],
        "netlist": [
            {
                "source_id": n.source_id,
                "target_id": n.target_id,
                "weight": n.weight,
            }
            for n in obs.netlist
        ],
        "current_metrics": {
            "wirelength": round(obs.current_wirelength, 4),
            "area_utilisation": round(obs.current_area_util, 4),
            "thermal_max": round(obs.current_thermal_max, 4),
        },
    }
    return json.dumps(data, indent=2)


def _extract_json(text: str) -> Optional[dict]:
    """Extract the first JSON object from an LLM response string."""
    # Try raw parse first
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # Extract from markdown code block
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Find first { ... } block
    match = re.search(r"\{[^{}]+\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return None


# ---------------------------------------------------------------------------
# LLM backend abstraction
# ---------------------------------------------------------------------------

class LLMAgent:
    """
    Thin wrapper around an LLM API.
    Configure via environment variables:
      LLM_PROVIDER  = "anthropic" | "openai" | "dummy"
      LLM_MODEL     = e.g. "claude-sonnet-4-20250514" or "gpt-4o"
      LLM_API_KEY   = your API key (or set ANTHROPIC_API_KEY / OPENAI_API_KEY)
    """

    def __init__(self):
        self.provider = os.environ.get("LLM_PROVIDER", "dummy").lower()
        self.model    = os.environ.get("LLM_MODEL", "claude-sonnet-4-20250514")
        self.api_key  = os.environ.get("LLM_API_KEY") or os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY")
        self.scan_step = float(os.environ.get("PPA_SCAN_STEP", "0.05"))
        self.bounds_epsilon = float(os.environ.get("PPA_BOUNDS_EPSILON", "1e-6"))

    def generate(self, user_message: str) -> str:
        if self.provider == "anthropic":
            return self._call_anthropic(user_message)
        elif self.provider == "openai":
            return self._call_openai(user_message)
        else:
            return self._dummy_response(user_message)

    def _call_anthropic(self, user_message: str) -> str:
        import anthropic
        client = anthropic.Anthropic(api_key=self.api_key)
        msg = client.messages.create(
            model=self.model,
            max_tokens=256,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )
        return msg.content[0].text

    def _call_openai(self, user_message: str) -> str:
        from openai import OpenAI
        base_url = os.environ.get("LLM_BASE_URL", None)
        client = OpenAI(api_key=self.api_key, base_url=base_url) if base_url else OpenAI(api_key=self.api_key)
        resp = client.chat.completions.create(
            model=self.model,
            max_tokens=256,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_message},
            ],
        )
        return resp.choices[0].message.content

    def _dummy_response(self, user_message: str) -> str:
        """
        Greedy heuristic fallback when no LLM is configured.
        Places the next block in the first available top-left gap.
        """
        import json as _json
        try:
            data = _json.loads(user_message)
        except Exception:
            return '{"block_id":"unknown","x":0.0,"y":0.0,"rotation":0}'

        remaining = data.get("remaining_blocks", [])
        placed    = data.get("placed_blocks", [])

        if not remaining:
            return '{"block_id":"","x":0.0,"y":0.0,"rotation":0}'

        block = remaining[0]
        bid   = block["block_id"]
        bw    = block["width"]
        bh    = block["height"]

        # Simple left-to-right, top-to-bottom packing
        occupied = [(b["x"], b["y"], b["x"] + b["width"], b["y"] + b["height"]) for b in placed]

        step = self.scan_step
        x, y = 0.0, 0.0
        placed_it = False
        while y + bh <= 1.0 + self.bounds_epsilon:
            while x + bw <= 1.0 + self.bounds_epsilon:
                ok = all(
                    x + bw <= ox0 or ox1 <= x or y + bh <= oy0 or oy1 <= y
                    for ox0, oy0, ox1, oy1 in occupied
                )
                if ok:
                    placed_it = True
                    break
                x = round(x + step, 4)
            if placed_it:
                break
            x = 0.0
            y = round(y + step, 4)

        if not placed_it:
            x, y = 0.0, 0.0

        return _json.dumps({"block_id": bid, "x": x, "y": y, "rotation": 0})


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_agent: Optional[LLMAgent] = None


def get_agent() -> LLMAgent:
    global _agent
    if _agent is None:
        _agent = LLMAgent()
    return _agent


def llm_act(obs: PPAObservation, max_retries: int = 5) -> PPAAction:
    """
    Given a PPAObservation, call the LLM and return a parsed PPAAction.
    Retries with failure feedback if the LLM returns an illegal position.
    """
    if not obs.remaining_blocks:
        return PPAAction(block_id="", x=0.0, y=0.0, rotation=0)

    agent       = get_agent()
    user_prompt = _build_user_prompt(obs)
    
    failed_positions = []
    
    for attempt in range(max_retries):
        # Add failed position history to prompt
        if failed_positions:
            feedback = "\n\nPREVIOUS FAILED ATTEMPTS (do NOT repeat these):\n"
            for fp in failed_positions:
                feedback += f"  - block={fp['block_id']} x={fp['x']} y={fp['y']} ILLEGAL\n"
            feedback += "Choose a DIFFERENT x,y position than all above.\n"
            prompt = user_prompt + feedback
        else:
            prompt = user_prompt

        raw_text = agent.generate(prompt)
        parsed   = _extract_json(raw_text)

        if parsed and "block_id" in parsed:
            action = PPAAction.from_dict(parsed)
            action.metadata["raw_llm_output"] = raw_text
            action.metadata["attempt"] = attempt + 1
            
            # Quick bounds check before returning
            block = next((b for b in obs.remaining_blocks if b.block_id == parsed["block_id"]), None)
            if block:
                x, y = parsed.get("x", 0), parsed.get("y", 0)
                if x + block.width <= 1.0 + agent.bounds_epsilon and y + block.height <= 1.0 + agent.bounds_epsilon:
                    return action
                else:
                    # Out of bounds — add to failed and retry
                    failed_positions.append({"block_id": parsed["block_id"], "x": round(x,3), "y": round(y,3)})
                    continue
            return action

        failed_positions.append({"block_id": "unknown", "x": -1, "y": -1})

    # All retries exhausted — use dummy fallback
    fallback_block = obs.remaining_blocks[0]
    dummy_response = agent._dummy_response(user_prompt)
    dummy_parsed   = _extract_json(dummy_response)
    if dummy_parsed:
        return PPAAction.from_dict(dummy_parsed)
    
    return PPAAction(
        block_id=fallback_block.block_id,
        x=0.0,
        y=0.0,
        rotation=0,
        metadata={"fallback": True},
    )