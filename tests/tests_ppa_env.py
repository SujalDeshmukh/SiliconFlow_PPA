"""
SiliconFlow-PPA: Test Suite

Tests:
  - Reward functions (unit)
  - Environment logic (unit)
  - API endpoints (integration via TestClient)
"""
from __future__ import annotations
import sys
import os

# Ensure src is importable
HERE = os.path.dirname(__file__)
SRC  = os.path.join(HERE, "../src")
sys.path.insert(0, os.path.abspath(SRC))

import pytest
from fastapi.testclient import TestClient

from envs.silicon_flow_ppa.models import (
    LogicBlock, NetConnection, PlacedBlock,
    PPAAction, PPAObservation,
)
from envs.silicon_flow_ppa.rewards import (
    check_legality,
    area_efficiency_reward,
    wirelength_reward,
    thermal_reward,
    compute_step_reward,
)
from envs.silicon_flow_ppa.server.environment import SiliconFlowPPAEnvironment
from envs.silicon_flow_ppa.server.app import app


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def env():
    return SiliconFlowPPAEnvironment(scenario="small_4block", max_steps=20)


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def two_blocks():
    return [
        PlacedBlock("a", 0.0, 0.0, 0.3, 0.3, 2.0),
        PlacedBlock("b", 0.5, 0.5, 0.2, 0.2, 1.0),
    ]


# ---------------------------------------------------------------------------
# Reward unit tests
# ---------------------------------------------------------------------------

class TestLegalityCheck:
    def test_legal_placement(self):
        existing = [PlacedBlock("a", 0.0, 0.0, 0.3, 0.3, 2.0)]
        new = PlacedBlock("b", 0.4, 0.0, 0.3, 0.3, 1.0)
        legal, penalty = check_legality(new, existing)
        assert legal is True
        assert penalty == 0.0

    def test_overlap_detected(self):
        existing = [PlacedBlock("a", 0.0, 0.0, 0.5, 0.5, 2.0)]
        new = PlacedBlock("b", 0.3, 0.3, 0.3, 0.3, 1.0)
        legal, penalty = check_legality(new, existing)
        assert legal is False
        assert penalty < 0.0

    def test_out_of_bounds(self):
        new = PlacedBlock("a", 0.8, 0.8, 0.4, 0.4, 1.0)
        legal, penalty = check_legality(new, [])
        assert legal is False
        assert penalty < 0.0

    def test_touching_but_not_overlapping(self):
        existing = [PlacedBlock("a", 0.0, 0.0, 0.3, 0.3, 2.0)]
        new = PlacedBlock("b", 0.3, 0.0, 0.3, 0.3, 1.0)
        legal, _ = check_legality(new, existing)
        assert legal is True


class TestAreaEfficiency:
    def test_empty(self):
        assert area_efficiency_reward([]) == 0.0

    def test_single_block_compact(self, two_blocks):
        # single tiny block in corner
        blocks = [PlacedBlock("a", 0.0, 0.0, 0.1, 0.1, 1.0)]
        score = area_efficiency_reward(blocks)
        assert 0.98 < score <= 1.0  # very compact

    def test_spread_blocks_lower_score(self):
        blocks = [
            PlacedBlock("a", 0.0, 0.0, 0.1, 0.1, 1.0),
            PlacedBlock("b", 0.9, 0.9, 0.1, 0.1, 1.0),
        ]
        score = area_efficiency_reward(blocks)
        assert score < 0.2  # bounding box ~ full die


class TestWirelength:
    def test_empty(self):
        assert wirelength_reward([], []) == 0.0

    def test_adjacent_blocks_high_reward(self):
        blocks = [
            PlacedBlock("a", 0.0, 0.0, 0.2, 0.2, 1.0),
            PlacedBlock("b", 0.2, 0.0, 0.2, 0.2, 1.0),
        ]
        net = [NetConnection("a", "b", 1.0)]
        score = wirelength_reward(blocks, net)
        assert score > 0.5

    def test_distant_blocks_low_reward(self):
        blocks = [
            PlacedBlock("a", 0.0, 0.0, 0.1, 0.1, 1.0),
            PlacedBlock("b", 0.9, 0.9, 0.1, 0.1, 1.0),
        ]
        net = [NetConnection("a", "b", 1.0)]
        score = wirelength_reward(blocks, net)
        assert score < 0.3


class TestThermal:
    def test_empty_no_hotspot(self):
        r, max_d = thermal_reward([])
        assert r == 1.0
        assert max_d == 0.0

    def test_high_power_cluster_penalised(self):
        # Three high-power blocks in the same corner
        blocks = [
            PlacedBlock("a", 0.0, 0.0, 0.2, 0.2, 10.0),
            PlacedBlock("b", 0.0, 0.2, 0.2, 0.2, 10.0),
            PlacedBlock("c", 0.2, 0.0, 0.2, 0.2, 10.0),
        ]
        r, max_d = thermal_reward(blocks)
        assert r < 1.0  # penalised
        assert max_d > 0.0

    def test_spread_power_ok(self):
        blocks = [
            PlacedBlock("a", 0.0, 0.0, 0.2, 0.2, 1.0),
            PlacedBlock("b", 0.8, 0.8, 0.2, 0.2, 1.0),
        ]
        r, _ = thermal_reward(blocks)
        assert r == 1.0


# ---------------------------------------------------------------------------
# Environment unit tests
# ---------------------------------------------------------------------------

class TestEnvironment:
    def test_reset_returns_all_blocks(self, env):
        obs = env.reset()
        assert isinstance(obs, PPAObservation)
        assert len(obs.remaining_blocks) == 4   # small_4block
        assert len(obs.placed_blocks) == 0
        assert obs.done is False

    def test_step_places_block(self, env):
        env.reset()
        obs = env.step(PPAAction(block_id="cpu", x=0.0, y=0.0, rotation=0))
        assert len(obs.placed_blocks) == 1
        assert obs.placed_blocks[0].block_id == "cpu"
        assert len(obs.remaining_blocks) == 3

    def test_illegal_step_does_not_place(self, env):
        env.reset()
        # Place cpu at (0,0)
        env.step(PPAAction(block_id="cpu", x=0.0, y=0.0, rotation=0))
        # Try to place another block in the exact same spot
        obs = env.step(PPAAction(block_id="gpu", x=0.0, y=0.0, rotation=0))
        # gpu should NOT be placed
        placed_ids = [b.block_id for b in obs.placed_blocks]
        assert "gpu" not in placed_ids
        assert obs.is_legal is False
        assert obs.overlap_penalty < 0

    def test_unknown_block_id_penalised(self, env):
        env.reset()
        obs = env.step(PPAAction(block_id="nonexistent", x=0.0, y=0.0))
        assert obs.is_legal is False
        assert "error" in obs.metadata

    def test_done_when_all_placed(self, env):
        env.reset()
        # Place all 4 blocks without overlap
        positions = [(0.0, 0.0), (0.4, 0.0), (0.0, 0.4), (0.6, 0.4)]
        block_ids = ["cpu", "gpu", "mem", "io"]
        obs = None
        for bid, (x, y) in zip(block_ids, positions):
            obs = env.step(PPAAction(block_id=bid, x=x, y=y, rotation=0))
        assert obs.done is True

    def test_cumulative_reward_increases_for_good_placement(self, env):
        env.reset()
        obs = env.step(PPAAction(block_id="cpu", x=0.0, y=0.0, rotation=0))
        state = env.state
        assert state.cumulative_reward == pytest.approx(obs.reward, abs=1e-4)

    def test_state_tracks_step_count(self, env):
        env.reset()
        env.step(PPAAction(block_id="cpu", x=0.0, y=0.0))
        env.step(PPAAction(block_id="mem", x=0.5, y=0.0))
        assert env.state.step_count == 2

    def test_max_steps_terminates_episode(self):
        env = SiliconFlowPPAEnvironment(scenario="small_4block", max_steps=2)
        env.reset()
        env.step(PPAAction(block_id="cpu", x=0.0, y=0.0))
        obs = env.step(PPAAction(block_id="mem", x=0.5, y=0.0))
        assert obs.done is True


# ---------------------------------------------------------------------------
# API integration tests
# ---------------------------------------------------------------------------

class TestAPI:
    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_reset_endpoint(self, client):
        resp = client.post("/reset")
        assert resp.status_code == 200
        data = resp.json()
        assert "remaining_blocks" in data
        assert "placed_blocks" in data
        assert data["done"] is False

    def test_step_endpoint(self, client):
        client.post("/reset")
        resp = client.post("/step", json={"block_id": "cpu", "x": 0.0, "y": 0.0, "rotation": 0})
        assert resp.status_code == 200
        data = resp.json()
        assert "placed_blocks" in data
        assert data["is_legal"] is True

    def test_state_endpoint(self, client):
        client.post("/reset")
        resp = client.get("/state")
        assert resp.status_code == 200
        data = resp.json()
        assert "step_count" in data
        assert "total_blocks" in data

    def test_render_endpoint(self, client):
        client.post("/reset")
        client.post("/step", json={"block_id": "cpu", "x": 0.0, "y": 0.0, "rotation": 0})
        resp = client.get("/render")
        assert resp.status_code == 200
        assert "cpu" in resp.text

    def test_scenario_endpoint(self, client):
        resp = client.get("/scenario")
        assert resp.status_code == 200
        assert "scenario" in resp.json()

    def test_full_episode_via_api(self, client):
        client.post("/reset")
        placements = [
            {"block_id": "cpu", "x": 0.0,  "y": 0.0,  "rotation": 0},
            {"block_id": "gpu", "x": 0.35, "y": 0.0,  "rotation": 0},
            {"block_id": "mem", "x": 0.0,  "y": 0.35, "rotation": 0},
            {"block_id": "io",  "x": 0.7,  "y": 0.35, "rotation": 0},
        ]
        for p in placements:
            resp = client.post("/step", json=p)
            assert resp.status_code == 200
        data = resp.json()
        assert data["done"] is True