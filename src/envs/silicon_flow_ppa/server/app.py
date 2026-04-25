"""
SiliconFlow-PPA: FastAPI Server

Wraps SiliconFlowPPAEnvironment in a REST API following the OpenEnv spec.

Endpoints:
  POST /reset         → start new episode
  POST /step          → place one block
  GET  /state         → episode metadata
  GET  /health        → liveness probe
  GET  /render        → ASCII visualisation of current layout
"""
import os
import sys

# Ensure src is on the path when running with uvicorn
HERE = os.path.dirname(__file__)
SRC  = os.path.abspath(os.path.join(HERE, "../../../.."))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse

from core.env_server import create_fastapi_app
from envs.silicon_flow_ppa.server.environment import SiliconFlowPPAEnvironment
from envs.silicon_flow_ppa.server.renderer import render_ascii

# ---------------------------------------------------------------------------
# Initialise environment from environment variables (for Docker config)
# ---------------------------------------------------------------------------

SCENARIO    = os.environ.get("PPA_SCENARIO",    "small_4block")
MAX_STEPS   = int(os.environ.get("PPA_MAX_STEPS", "50"))
RANDOMISE   = os.environ.get("PPA_RANDOMISE",   "false").lower() == "true"

env = SiliconFlowPPAEnvironment(
    scenario=SCENARIO,
    max_steps=MAX_STEPS,
    randomise_scenario=RANDOMISE,
)

# Build base FastAPI app from the factory (provides /reset, /step, /state, /health)
app = create_fastapi_app(env)
app.title = "SiliconFlow-PPA Environment"
app.description = (
    "OpenEnv-compatible RL environment for chip physical design optimisation. "
    "The agent places logic blocks on a 2D die, scored on Area, Wirelength, and Thermal metrics."
)


# ---------------------------------------------------------------------------
# Extra endpoint: ASCII layout render
# ---------------------------------------------------------------------------

@app.get("/render", response_class=PlainTextResponse)
def render():
    """Return a human-readable ASCII art view of the current chip layout."""
    try:
        return render_ascii(env._placed_blocks, env._remaining_blocks, env.die_width, env.die_height)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Extra endpoint: scenario info
# ---------------------------------------------------------------------------

@app.get("/scenario")
def scenario_info():
    return {
        "scenario": env.scenario_name,
        "total_blocks": env._total_blocks,
        "placed_blocks": len(env._placed_blocks),
        "remaining_blocks": len(env._remaining_blocks),
        "max_steps": env.max_steps,
        "die": {"width": env.die_width, "height": env.die_height},
    }