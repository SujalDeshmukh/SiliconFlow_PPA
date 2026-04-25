"""
SiliconFlow-PPA: RL Training Loop.

Runs the LLM agent through multiple episodes, collecting
(observation, action, reward) tuples and logging metrics.

Usage:
  # Without Docker (in-process server):
  python scripts/train.py --scenario small_4block --episodes 10 --provider dummy

  # Against a running Docker container:
  python scripts/train.py --scenario medium_6block --episodes 50 \
      --server-url http://localhost:8000 --provider anthropic
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from typing import List

HERE = os.path.dirname(__file__)
SRC  = os.path.abspath(os.path.join(HERE, "../src"))
sys.path.insert(0, SRC)

from envs.silicon_flow_ppa.server.environment import SiliconFlowPPAEnvironment
from envs.silicon_flow_ppa.models import PPAAction, PPAObservation
from envs.silicon_flow_ppa.inference import llm_act


# ---------------------------------------------------------------------------
# Metrics tracker
# ---------------------------------------------------------------------------

class EpisodeMetrics:
    def __init__(self, episode_id: int):
        self.episode_id = episode_id
        self.steps = 0
        self.total_reward = 0.0
        self.illegal_placements = 0
        self.final_wirelength = 0.0
        self.final_area_util = 0.0
        self.final_thermal = 0.0
        self.completed = False    # all blocks placed within budget

    def update(self, obs: PPAObservation):
        self.steps += 1
        self.total_reward += obs.reward or 0.0
        if not obs.is_legal:
            self.illegal_placements += 1
        self.final_wirelength = obs.current_wirelength
        self.final_area_util  = obs.current_area_util
        self.final_thermal    = obs.current_thermal_max
        if obs.done and len(obs.remaining_blocks) == 0:
            self.completed = True

    def to_dict(self) -> dict:
        return {
            "episode":           self.episode_id,
            "steps":             self.steps,
            "total_reward":      round(self.total_reward, 4),
            "illegal":           self.illegal_placements,
            "wirelength":        round(self.final_wirelength, 4),
            "area_utilisation":  round(self.final_area_util, 4),
            "thermal_max":       round(self.final_thermal, 4),
            "completed":         self.completed,
        }


# ---------------------------------------------------------------------------
# Single episode runner
# ---------------------------------------------------------------------------

def run_episode(
    env: SiliconFlowPPAEnvironment,
    episode_id: int,
    verbose: bool = False,
) -> EpisodeMetrics:
    obs = env.reset()
    metrics = EpisodeMetrics(episode_id)

    if verbose:
        print(f"\n{'='*60}")
        print(f"Episode {episode_id} | Scenario: {env.scenario_name}")
        print(f"Blocks to place: {[b.block_id for b in obs.remaining_blocks]}")
        print(f"{'='*60}")

    while not obs.done:
        action = llm_act(obs)

        if verbose:
            print(f"  Step {metrics.steps+1:02d} | Place '{action.block_id}' "
                  f"at ({action.x:.3f}, {action.y:.3f}) rot={action.rotation}")

        obs = env.step(action)
        metrics.update(obs)

        if verbose:
            legal_str = "✓" if obs.is_legal else "✗ ILLEGAL"
            print(f"         reward={obs.reward:.3f}  {legal_str}  "
                  f"wl={obs.current_wirelength:.3f}  "
                  f"area={obs.current_area_util:.3f}  "
                  f"therm={obs.current_thermal_max:.3f}")

    if verbose:
        print(f"\n  Episode done | total_reward={metrics.total_reward:.3f}  "
              f"completed={metrics.completed}  illegal={metrics.illegal_placements}")

    return metrics


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(
    scenario: str,
    num_episodes: int,
    max_steps: int,
    provider: str,
    verbose: bool,
    log_file: str,
):
    os.environ.setdefault("LLM_PROVIDER", provider)

    env = SiliconFlowPPAEnvironment(
        scenario=scenario,
        max_steps=max_steps,
        randomise_scenario=(scenario == "random"),
    )

    all_metrics: List[EpisodeMetrics] = []
    print(f"\n🚀  SiliconFlow-PPA Training")
    print(f"    Scenario  : {scenario}")
    print(f"    Episodes  : {num_episodes}")
    print(f"    Max steps : {max_steps}")
    print(f"    LLM       : {provider}\n")

    for ep in range(1, num_episodes + 1):
        t0 = time.time()
        metrics = run_episode(env, ep, verbose=verbose)
        elapsed = time.time() - t0
        all_metrics.append(metrics)

        if not verbose:
            status = "✓" if metrics.completed else "✗"
            print(f"  Ep {ep:03d}/{num_episodes}  {status}  "
                  f"reward={metrics.total_reward:6.3f}  "
                  f"steps={metrics.steps:2d}  "
                  f"illegal={metrics.illegal_placements}  "
                  f"wl={metrics.final_wirelength:.3f}  "
                  f"({elapsed:.1f}s)")

    # Summary
    completed = sum(1 for m in all_metrics if m.completed)
    avg_reward = sum(m.total_reward for m in all_metrics) / len(all_metrics)
    avg_wl     = sum(m.final_wirelength for m in all_metrics) / len(all_metrics)
    print(f"\n{'='*60}")
    print(f"  Summary over {num_episodes} episodes")
    print(f"  Completion rate : {completed}/{num_episodes} ({100*completed//num_episodes}%)")
    print(f"  Avg total reward: {avg_reward:.4f}")
    print(f"  Avg wirelength  : {avg_wl:.4f}")
    print(f"{'='*60}\n")

    # Write JSONL log
    if log_file:
        os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
        with open(log_file, "w") as f:
            for m in all_metrics:
                f.write(json.dumps(m.to_dict()) + "\n")
        print(f"  Log saved → {log_file}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SiliconFlow-PPA RL Training Loop")
    parser.add_argument("--scenario",  default="small_4block",
                        choices=["small_4block", "medium_6block", "hard_8block", "random"])
    parser.add_argument("--episodes",  type=int,  default=5)
    parser.add_argument("--max-steps", type=int,  default=50)
    parser.add_argument("--provider",  default="dummy",
                        choices=["dummy", "anthropic", "openai"],
                        help="LLM backend to use")
    parser.add_argument("--verbose",   action="store_true")
    parser.add_argument("--log-file",  default="logs/training.jsonl")
    args = parser.parse_args()

    train(
        scenario=args.scenario,
        num_episodes=args.episodes,
        max_steps=args.max_steps,
        provider=args.provider,
        verbose=args.verbose,
        log_file=args.log_file,
    )