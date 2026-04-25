# SiliconFlow-PPA

**OpenEnv RL Environment for Chip Physical Design Optimisation**

An LLM-powered autonomous hardware engineer that places logic blocks on a 2D chip die, optimising for **Power, Performance, and Area (PPA)** metrics.

---

## What it does

The agent receives a virtual chip die, a list of logic blocks (with sizes and power ratings), and a netlist defining connectivity. It places blocks one at a time, receiving reward signals based on:

| Metric | What it measures |
|--------|-----------------|
| **Area Efficiency** | Compactness of bounding box |
| **Wirelength** | Total Manhattan distance of netlist connections |
| **Thermal Density** | Power hotspot detection |
| **Legality** | No overlaps, no out-of-bounds placements |
| **Completion** | Bonus for placing all blocks within budget |

---

## Project Structure

```
silicon_flow_ppa/
├── src/
│   ├── core/
│   │   ├── env_server.py          # Abstract base classes + FastAPI factory
│   │   └── http_env_client.py     # Generic HTTP client base
│   └── envs/
│       └── silicon_flow_ppa/
│           ├── models.py           # PPAAction, PPAObservation, PPAState
│           ├── rewards.py          # Composable reward functions
│           ├── client.py           # SiliconFlowPPAClient (HTTP)
│           ├── inference.py        # LLM agent + prompt engineering
│           └── server/
│               ├── environment.py  # Core placement logic
│               ├── app.py          # FastAPI app (endpoints)
│               ├── renderer.py     # ASCII layout visualiser
│               └── Dockerfile
├── tests/
│   └── test_ppa_env.py            # Pytest suite (unit + integration)
├── scripts/
│   ├── quickstart.py              # No-Docker demo
│   └── train.py                   # RL training loop
├── docker-compose.yml
└── requirements.txt
```

---

## Quick Start (no Docker)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run demo episode with the built-in greedy agent
python scripts/quickstart.py

# 3. Run tests
cd src && python -m pytest ../tests/ -v
```

---

## Run with Docker

```bash
# Build and start the environment server
docker compose up --build

# The server is live at http://localhost:8000
# Check it:
curl http://localhost:8000/health

# Start a new episode:
curl -X POST http://localhost:8000/reset

# Place a block:
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"block_id":"cpu","x":0.1,"y":0.1,"rotation":0}'

# View ASCII layout:
curl http://localhost:8000/render
```

---

## Run the Training Loop

```bash
# Dummy agent (built-in greedy heuristic, no API key needed)
python scripts/train.py --scenario small_4block --episodes 10 --provider dummy --verbose

# Anthropic Claude agent
export ANTHROPIC_API_KEY=sk-ant-...
python scripts/train.py --scenario medium_6block --episodes 50 --provider anthropic

# Hard scenario
python scripts/train.py --scenario hard_8block --episodes 100 --provider anthropic --log-file logs/hard.jsonl
```

---

## API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Liveness probe |
| `/reset` | POST | Start new episode |
| `/step` | POST | Place one block |
| `/state` | GET | Episode metadata |
| `/render` | GET | ASCII layout view |
| `/scenario` | GET | Scenario info |

### Step payload
```json
{
  "block_id": "cpu",
  "x": 0.1,
  "y": 0.1,
  "rotation": 0
}
```

### Observation response
```json
{
  "die_width": 1.0,
  "die_height": 1.0,
  "remaining_blocks": [...],
  "placed_blocks": [...],
  "netlist": [...],
  "step_reward": 4.12,
  "is_legal": true,
  "current_wirelength": 0.342,
  "current_area_util": 0.78,
  "current_thermal_max": 0.31,
  "done": false,
  "reward": 4.12
}
```

---

## Scenarios

| Name | Blocks | Difficulty |
|------|--------|------------|
| `small_4block` | CPU, GPU, Mem, IO | Easy |
| `medium_6block` | 2× CPU, GPU, Cache, PCIe, DDR | Medium |
| `hard_8block` | A53×2, A78, Mali, NPU, ISP, Modem, PMU | Hard |

---

## Reward Function

```
R = 2.0 × legality
  + 1.0 × area_efficiency
  + 1.0 × wirelength
  + 1.5 × thermal
  + overlap_penalty        (−5 per illegal step)
  + 3.0 × completion_bonus (only at episode end, if all blocks placed)
```

---

## LLM Configuration

Set environment variables before training:

| Variable | Values | Description |
|----------|--------|-------------|
| `LLM_PROVIDER` | `dummy`, `anthropic`, `openai` | Backend |
| `LLM_MODEL` | e.g. `claude-sonnet-4-20250514` | Model name |
| `ANTHROPIC_API_KEY` | `sk-ant-...` | Anthropic key |
| `OPENAI_API_KEY` | `sk-...` | OpenAI key |