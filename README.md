---
title: SiliconFlow-PPA
emoji: 🔬
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---
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
| **Routing Congestion (proxy)** | Grid demand concentration |
| **Timing Cost (proxy)** | Weighted long-net pressure |
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

# Groq model through OpenAI-compatible API (Windows cmd/powershell style)
set LLM_API_KEY=gsk_your_actual_groq_key
set LLM_BASE_URL=https://api.groq.com/openai/v1/
set LLM_MODEL=llama-3.1-8b-instant
set LLM_PROVIDER=openai
python scripts/train.py --scenario medium_6block --episodes 50 --provider openai

# Hard scenario
python scripts/train.py --scenario hard_8block --episodes 100 --provider openai --log-file logs/hard.jsonl
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
  + 0.8 × congestion_proxy
  + 0.8 × timing_proxy
  + overlap_penalty        (−5 per illegal step)
  + 3.0 × completion_bonus (only at episode end, if all blocks placed)
```

---

## Configuration (No Hardcoded Knobs)

Runtime behavior is now configurable with files + env vars:

- `configs/scenarios.json`: block catalogs and netlists
- `configs/reward_config.json`: reward weights and metric constants
- `PPA_SCENARIOS_FILE`: override scenario file path
- `PPA_REWARD_CONFIG`: override reward config path
- `PPA_SERVER_URL`: server endpoint for HTTP client
- `PPA_SCAN_STEP`: greedy fallback scan step size
- `PPA_BOUNDS_EPSILON`: numeric bounds tolerance
- `OPENENV_BASE_URL`: endpoint used by `openenv.yaml`

---

## LLM Configuration

Set environment variables before training:

| Variable | Values | Description |
|----------|--------|-------------|
| `LLM_PROVIDER` | `dummy`, `openai` | Backend (`openai` path is also used for Groq/OpenAI-compatible APIs) |
| `LLM_MODEL` | e.g. `llama-3.1-8b-instant` | Model name |
| `LLM_API_KEY` | `gsk-...` or provider key | Primary API key variable |
| `LLM_BASE_URL` | e.g. `https://api.groq.com/openai/v1/` | OpenAI-compatible endpoint override |
| `OPENAI_API_KEY` | `sk-...` (optional fallback) | Secondary key fallback supported by code |

Example (your setup):

```bash
set LLM_API_KEY=gsk_your_actual_groq_key
set LLM_BASE_URL=https://api.groq.com/openai/v1/
set LLM_MODEL=llama-3.1-8b-instant
set LLM_PROVIDER=openai
```

---

## Training Evidence for Judging

To satisfy "show real training end-to-end" and "show improvement in rewards", include:

- baseline run log (Groq initial/before training behavior)
- trained run log (SmolLM2 from Google Colab, evaluated on same scenario)
- same scenario + same episode count for both (recommend 50)
- comparison plots from `scripts/plot_metrics.py`
- at least one terminal training screenshot (Groq) and one Colab screenshot (SmolLM2)

### Submission-time status (current)

This submission already includes verified real-run artifacts from this repository:
- logs in `logs/*.jsonl`
- plots in `plots/*.png`

We are submitting these committed results now for deadline safety.

### During-training artifacts

Alongside numeric metrics, include:
- terminal screenshots from Groq training
- Google Colab screenshots from SmolLM2 training
- any training dataset snapshots/exports used during training

Screenshot placeholders:

```md
![Groq terminal training run](assets/groq_training_terminal.png)
![SmolLM2 training in Google Colab](assets/smollm2_colab_training.png)
```

This directly addresses the 20% criterion: observable evidence of learning progress.

---

## Real EDA Starter (Phase 1 + 2)

This repository now includes starter interfaces for real tool-driven evaluation:

- `src/eda_flow/models.py`: canonical metric schema (`EDAMetrics`) + scoring
- `src/eda_flow/parsers/openroad_reports.py`: OpenROAD text/report parser
- `src/eda_flow/adapters/openroad_runner.py`: stage runner with logs + artifacts
- `src/eda_flow/orchestrator.py`: run orchestration + result persistence
- `scripts/run_eda_eval.py`: CLI entrypoint for end-to-end flow evaluation

Run the starter pipeline:

```bash
python scripts/run_eda_eval.py --design-name my_block
```

With custom stages:

```bash
python scripts/run_eda_eval.py \
  --design-name my_block \
  --stage "floorplan::openroad -exit scripts/floorplan.tcl::reports/floorplan/*.rpt,reports/floorplan/*.log" \
  --stage "route::openroad -exit scripts/route.tcl::reports/route/*.rpt,reports/route/*.log"
```

Or run from a reusable stage config:

```bash
python scripts/run_eda_eval.py --stage-config configs/openroad_stages.example.json
```

Generated outputs are saved under `artifacts/eda_runs/<run_id>/` as:

- `evaluation.json` (all parsed metrics + per-stage results)
- `score.json` (unified objective score breakdown)

To connect a real OpenROAD flow:

- Copy `configs/openroad_stages.example.json` to your own config (for your design).
- Point each stage `command` to your actual `.tcl` flow scripts.
- Ensure stage scripts emit reports under the matching `report_globs`.
- Run with `--stage-config` and inspect `evaluation.json` + `score.json`.

Ready-to-edit template provided:

- `configs/openroad_stages.my_design.json`
- `EDA_SETUP.md` (step-by-step first real run guide)