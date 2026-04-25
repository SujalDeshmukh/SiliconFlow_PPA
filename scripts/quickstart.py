"""
SiliconFlow-PPA: Quick-start demo (no Docker needed).

Runs one episode with the dummy greedy agent and prints a live
ASCII render of the chip layout after each placement.

Usage:
  python scripts/quickstart.py
"""
from __future__ import annotations
import os
import sys

HERE = os.path.dirname(__file__)
SRC  = os.path.abspath(os.path.join(HERE, "../src"))
sys.path.insert(0, SRC)

os.environ["LLM_PROVIDER"] = "dummy"

from envs.silicon_flow_ppa.server.environment import SiliconFlowPPAEnvironment
from envs.silicon_flow_ppa.server.renderer import render_ascii
from envs.silicon_flow_ppa.inference import llm_act


def main():
    print("=" * 60)
    print("  SiliconFlow-PPA  |  Quick-start Demo")
    print("=" * 60)

    env = SiliconFlowPPAEnvironment(scenario="small_4block", max_steps=20)
    obs = env.reset()

    print(f"\nScenario : small_4block")
    print(f"Die size : {env.die_width} × {env.die_height} (normalised)")
    print(f"Blocks   : {[b.block_id for b in obs.remaining_blocks]}\n")

    step = 0
    while not obs.done:
        action = llm_act(obs)
        obs = env.step(action)
        step += 1

        print(f"Step {step:02d} | placed '{action.block_id}' "
              f"at ({action.x:.3f}, {action.y:.3f})  "
              f"reward={obs.reward:.3f}  "
              f"legal={obs.is_legal}")

    print("\n" + render_ascii(obs.placed_blocks, obs.remaining_blocks))

    state = env.state
    print(f"\n{'='*60}")
    print(f"  Episode complete!")
    print(f"  Steps taken        : {state.step_count}")
    print(f"  Cumulative reward  : {state.cumulative_reward:.4f}")
    print(f"  Blocks placed      : {state.placed_count}/{state.total_blocks}")
    print(f"  Final wirelength   : {obs.current_wirelength:.4f}")
    print(f"  Final area util    : {obs.current_area_util:.4f}")
    print(f"  Thermal max        : {obs.current_thermal_max:.4f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()