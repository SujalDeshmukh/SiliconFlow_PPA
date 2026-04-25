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

SYSTEM_PROMPT = """You are SiliconFlow-PPA, an expert hardware engineer specialising
in chip physical design. Your task is to place logic blocks on a 2D chip die to
minimise wirelength, maximise area efficiency, and avoid thermal hotspots.

Die dimensions are normalised to [0, 1] × [0, 1].

You will receive the current layout state as JSON. You must respond with ONLY a
valid JSON object specifying your placement decision:

{
  "block_id": "<id of the block to place>",
  "x": <float, top-left x in [0, 1]>,
  "y": <float, top-left y in [0, 1]>,
  "rotation": <0 for no rotation, 1 for 90 degrees>
}

Rules:
1. Place only blocks listed in "remaining_blocks".
2. The block must fit entirely within the die (x + width <= 1, y + height <= 1).
3. The block must NOT overlap any already placed block.
4. High-power blocks placed adjacent to each other create thermal hotspots — avoid this.
5. Minimise the total Manhattan distance between connected blocks (see "netlist").
6. Output ONLY the JSON object — no explanation, no markdown, no extra text.
"""


def _build_user_prompt(obs: PPAObservation) -> str:
    """Serialise the current observation into a concise prompt for the LLM."""
    data = {
        "die": {"width": obs.die_width, "height": obs.die_height},
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
        client = OpenAI(api_key=self.api_key)
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

        step = 0.05
        x, y = 0.0, 0.0
        placed_it = False
        while y + bh <= 1.0 + 1e-6:
            while x + bw <= 1.0 + 1e-6:
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


def llm_act(obs: PPAObservation) -> PPAAction:
    """
    Given a PPAObservation, call the LLM and return a parsed PPAAction.
    Falls back to a random legal placement if the LLM output is unparseable.
    """
    if not obs.remaining_blocks:
        return PPAAction(block_id="", x=0.0, y=0.0, rotation=0)

    agent       = get_agent()
    user_prompt = _build_user_prompt(obs)
    raw_text    = agent.generate(user_prompt)
    parsed      = _extract_json(raw_text)

    if parsed and "block_id" in parsed:
        action = PPAAction.from_dict(parsed)
        action.metadata["raw_llm_output"] = raw_text
        return action

    # Fallback: place first remaining block at (0, 0)
    fallback_block = obs.remaining_blocks[0]
    return PPAAction(
        block_id=fallback_block.block_id,
        x=0.0,
        y=0.0,
        rotation=0,
        metadata={"fallback": True, "raw_llm_output": raw_text},
    )