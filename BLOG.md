# SiliconFlow-PPA: Teaching LLMs to Design Chips with Reinforcement Learning

> An autonomous hardware engineer that learns to place CPU, GPU, RAM, and I/O blocks on a 2D die — optimising Power, Performance, and Area one step at a time.

| Domain | Framework | Backends | Stack |
|--------|-----------|----------|-------|
| ASIC / EDA / RL | OpenEnv + FastAPI | Groq (OpenAI-compatible) · OpenAI-compatible · Dummy | Python · Docker · Pydantic |

---

## 01  Why This Matters — Chip Placement is a $100B Bottleneck

Everything from your phone's processor to your graphics card, from your computer's CPU to the server-grade CPUs used in data centres have had their origins as a **floorplan**. Before any transistors are made at all, the decision needs to be made of where on the silicon wafer the macro blocks are to go. This task is known as **macro/block placement**.

With today's heterogeneous SoCs, a single chip could include a CPU cluster, a GPU, an NPU, ISP, modem, and memory controller. Placing these blocks incorrectly will cause you:

- 🔥 **Thermal Hotspots** - Having high-power blocks clustered together will create thermal hotspots that will cause poor performance and shortened lifespan for the chip.
- 📏 **Wasted Wirelength** - Placement problems may require longer wires, which means added capacitance, more latency, and wasted energy.
- 🗺️ **Routing Issues** - Clustering blocks will lead to routing congestion, causing parts of the layout to fail DRC. This sends the entire layout back to square one.
- ⏱️ **Missed Timing** - Critical paths spanning the layout will miss their timing targets, leading to costly spins.

Conventional EDA tools are powerful but deterministic and extremely time-consuming to iterate. They lack the capacity to *reason about trade-offs* the way a seasoned human engineer does.

> **The core insight:** Placement is not a one-shot decision problem — it is a sequential, multi-objective, partially-observable Markov Decision Process. Framing it as RL is not just elegant; it's correct.

---

## 02  Approach — RL Loop: From One-Shot Prompt to Learnable Control

The naive approach to LLM-driven chip placement is to ask the model to produce a complete floorplan in a single prompt. This fails for the same reason that humans don't design chips in one sketch: the search space is enormous, constraints are non-local, and every decision affects future options.

SiliconFlow-PPA reframes placement as an **episodic reinforcement learning loop**:

**Step 1 — Observe**
The agent receives the current die state: dimensions, all remaining and already-placed blocks (with sizes, power ratings), the netlist encoding connectivity, and running PPA proxy metrics. This is a rich, structured `PPAObservation` object.

**Step 2 — Act**
The LLM produces a single structured placement decision — which block to place, where (normalised x/y coordinates on the die), and at what rotation (0°, 90°, 180°, 270°). The action schema is strict:

```json
{"block_id": "cpu", "x": 0.10, "y": 0.05, "rotation": 0}
```

**Step 3 — Evaluate**
The environment checks legality (bounds + overlap), computes PPA proxy metrics — area efficiency, wirelength, thermal density, routing congestion, timing risk — and returns the updated state.

**Step 4 — Reward**
A weighted composite reward signal is computed and returned. The agent uses this to shape future decisions. The episode continues until all blocks are placed or the step budget is exhausted.

> **The key takeaway:** RL-style interaction turns chip placement from a one-shot prompt into a learnable control problem. The model receives feedback at every step and must reason about long-horizon consequences — exactly the kind of trade-off reasoning that makes human experts valuable.

---
## 03  Why This Approach Matters

Chip design is one of the most sophisticated engineering problems faced by humans. Designing a single cutting-edge SoC requires 500+ people working for several years. Every tool that helps speed up even one process along the way — even by 10% — has significant implications for the economy and technology.

SiliconFlow-PPA shows that **formal environment design**, **multi-objective reward shaping**, and **action selection by LLMs** are a promising direction in pursuing fully automated floorplanning. The model does not need to have any prior knowledge of chip design — it learns from feedback, similar to a young engineer from an experienced one.

The key point of this research is not that LLMs will do chip placement better than experts now. Every new episode brings more data, which allows us to improve reward shaping, and every improvement brings us closer to the ultimate solution. The engine keeps building momentum.

> **Interaction through RL turns chip placement into a controllable problem instead of solving it in one go through prompt design.** Getting the right formulation in terms of environment, observations, actions, and rewards allows turning an AI-hard problem into an easily solvable one. And this is what SiliconFlow-PPA is all about.
---

## 04  Implementation — System Architecture

SiliconFlow-PPA is a modular stack with clean separation of concerns:

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
│   └── test_ppa_env.py             # Pytest suite (unit + integration)
├── scripts/
│   ├── quickstart.py               # No-Docker demo
│   ├── train.py                    # RL training loop
│   └── plot_metrics.py             # Reward/metrics visualisation
├── docker-compose.yml
└── requirements.txt
```

### Environment Server

Built on **FastAPI** and `openenv-core`, the server exposes a clean REST API. Every endpoint is typed via Pydantic — there is no ambiguity in what gets sent or received. The server can run bare or inside Docker via the provided `docker-compose.yml`.

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Liveness probe — confirm the server is up before training |
| `/reset` | POST | Initialises a new episode; returns the first `PPAObservation` |
| `/step` | POST | Places one block; returns updated observation + reward + done flag |
| `/state` | GET | Current episode metadata (step count, placed blocks, running metrics) |
| `/render` | GET | ASCII art view of the current die layout — handy for debugging |
| `/scenario` | GET | Active scenario configuration (block list, die dimensions, netlist) |

### Typed Data Contracts

The three Pydantic models that flow through the system:

- `PPAAction` — block_id, x, y, rotation
- `PPAObservation` — full die state, remaining/placed blocks, netlist, running metrics, reward, done
- `PPAState` — episode metadata including step count and scenario config

### Inference Layer

The `inference.py` module builds a structured prompt containing the current die state, occupied regions, remaining blocks, and placement hints, then parses the model's JSON output. A retry loop handles invalid or illegal outputs — if the model produces a malformed action or places a block illegally, the error and the failed position are fed back as context for the next attempt. A deterministic `dummy` agent serves as a baseline when no API key is available.

---

## 05  Test Cases — Three Benchmark Scenarios

SiliconFlow-PPA ships with three bundled scenarios of increasing difficulty. Each is defined in JSON and can be overridden via environment variables or config file.

| Scenario | Difficulty | Blocks | Notes |
|----------|------------|--------|-------|
| `small_4block` | Easy | CPU · GPU · Memory · IO | Entry-level — good for validating reward logic and prompt engineering |
| `medium_6block` | Medium | 2× CPU · GPU · Cache · PCIe · DDR | Multi-core tension between cache locality and memory bus placement |
| `hard_8block` | Hard | A53×2 · A78 · Mali · NPU · ISP · Modem · PMU | Real SoC topology — thermal and congestion trade-offs are severe |

The hard scenario mirrors a realistic mobile SoC: the NPU and ISP are power-hungry and must be separated; the modem needs proximity to the PMU; the A53 and A78 clusters have shared cache dependencies. Getting all eight blocks placed legally and efficiently is non-trivial even for experienced human engineers.

---

## 06  Learning Signal — Reward Design

The most important engineering decision in any RL system is the reward function. SiliconFlow-PPA uses a **composable, weighted multi-objective reward**:

```
R =  2.0 × legality
  +  1.0 × area_efficiency
  +  1.0 × wirelength_quality
  +  1.5 × thermal_quality
  +  0.5 × congestion_quality
  +  0.5 × timing_quality
  −  5.0 × overlap_penalty      (per illegal step)
  +  3.0 × completion_bonus     (final legal episode only)
```

### Why each signal matters

| Signal | What it measures | Why it's included |
|--------|-----------------|-------------------|
| `legality` | No overlap · No out-of-bounds | Keeps the agent in valid design space. Violations are immediately penalised, teaching constraint respect early. |
| `area_efficiency` | Compactness of placed blocks' bounding box relative to die | Encourages tight, compact layouts that leave room for routing. |
| `wirelength_quality` | Total Manhattan distance of netlist connections between placed blocks | Minimising wire length directly reduces RC delay and dynamic power — the two most impactful PPA levers. |
| `thermal_quality` | Maximum thermal density proxy (power × proximity) | High-power blocks near each other create hotspots. This signal forces the agent to spatially distribute heat sources. |
| `congestion_quality` | Routing density estimate in the most congested grid cell | Prevents tight clusters that cause routing DRC failures downstream. |
| `timing_quality` | Proxy risk score for critical path crossings | Discourages placements that create long timing-critical paths across the die. |
| `overlap_penalty` | Hard penalty: −5 per illegal placement | Strong negative shaping to make illegal placements immediately and unambiguously costly. |
| `completion_bonus` | +3 at episode end if all blocks placed legally | Creates long-horizon pressure — the agent must complete the entire floorplan, not just place well locally. |

> **Design note:** The reward weights are configurable via a JSON reward config file. Researchers can favour area over wirelength, or increase thermal weight for power-constrained scenarios, without touching source code.

---

## 07  Environment Internals — How an Episode Works

Each episode in `SiliconFlowPPAEnvironment` is self-contained and requires no manual intervention once the loop starts.

```python
# Pseudo-code of a full episode
obs = client.reset()
while not done:
    action = agent.act(obs)          # LLM or dummy produces PPAAction
    obs, reward, done = client.step(action)
    logger.log(obs, reward, action)

# At episode end:
metrics = obs.cumulative_metrics    # area util, wirelength, thermal max, illegals
```
Each time `step(action)` is invoked, the following steps occur in the environment:

1. Resolve target block by `block_id`
2. Rotate the target block bounding box according to the rotation required
3. Legality check: first, check for bounds, followed by checking overlap against all placed blocks
4. If legal, place the block in `placed_blocks` and remove from `remaining_blocks`
5. Compute new values for all proxy metrics used by PPA
6. Calculate reward
7. Return `PPAObservation` + reward + done

### ASCII Renderer

The `/render` endpoint returns an ASCII art representation of the current die, annotated with block labels — invaluable for debugging without any GUI tooling:

```
+--------------------------------------------------+
| .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  . |
| .  [CPU][CPU][CPU]  .  .  [GPU][GPU][GPU][GPU]  . |
| .  [CPU][CPU][CPU]  .  .  [GPU][GPU][GPU][GPU]  . |
| .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  . |
| .  .  [MEM][MEM]  .  .  .  .  [IO]  .  .  .  . |
+--------------------------------------------------+
```

---

## 08  LLM Integration — Making LLMs Reliable for Spatial Reasoning

Spatial reasoning is notoriously difficult for text models. We implemented several practical guardrails:

### Strict Action Schema

Every action must conform to a four-field JSON schema. The model cannot produce free-form text — the inference layer extracts JSON from the model's output using a robust parser with retry logic.

### Structured Prompt

The prompt given to the model includes: die dimensions, a list of remaining blocks with their sizes and power ratings, a list of occupied rectangles (block_id, x1, y1, x2, y2), running metrics (current wirelength, thermal max, area utilisation), and placement hints based on the netlist (e.g., "CPU and Cache are connected — consider placing them close").

### Retry Loop with Feedback

If the model produces a malformed JSON response or requests an illegal placement, the error is appended to the prompt and the model is asked to try again. Failed positions are explicitly fed back: *"Your last attempt at (0.1, 0.1) overlapped with CPU. Try a different location."*

### Deterministic Fallback

The `dummy` provider implements a greedy heuristic that places blocks in netlist-sorted order at the next available legal position — a baseline for benchmarking without any API key.

### LLM Providers

| Provider | Notes |
|----------|-------|
|  **Groq (via OpenAI-compatible API)** | Primary backend used in our runs. Configure `LLM_PROVIDER=openai` and set `LLM_BASE_URL=https://api.groq.com/openai/v1/`. |
|  **OpenAI-compatible providers** | Any provider exposing OpenAI chat completions is supported via the same interface. |
|  **Dummy (Greedy)** | No API key needed. Greedy heuristic baseline for reproducible benchmarking. |

---

## 09  Running It Yourself — Quickstart & Training

### Without Docker (fastest path)

```bash
# Install dependencies
pip install -r requirements.txt

# Run a single demo episode (greedy agent, no API key needed)
python scripts/quickstart.py

# Run the test suite
cd src && python -m pytest ../tests/ -v
```

### With Docker

```bash
# Build and start the environment server
docker compose up --build

# Health check
curl http://localhost:8000/health

# Start a new episode
curl -X POST http://localhost:8000/reset

# Place a block
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"block_id":"cpu","x":0.1,"y":0.1,"rotation":0}'

# View the ASCII layout
curl http://localhost:8000/render
```

### Training Loop

```bash
# Greedy baseline — no API key needed
python scripts/train.py \
  --scenario small_4block \
  --episodes 10 \
  --provider dummy \
  --verbose

# Groq agent (OpenAI-compatible endpoint) — medium scenario
set LLM_API_KEY=gsk_your_actual_groq_key
set LLM_BASE_URL=https://api.groq.com/openai/v1/
set LLM_MODEL=llama-3.1-8b-instant
set LLM_PROVIDER=openai
python scripts/train.py \
  --scenario medium_6block \
  --episodes 50 \
  --provider openai

# Hard scenario with JSONL logging
python scripts/train.py \
  --scenario hard_8block \
  --episodes 100 \
  --provider openai \
  --log-file logs/hard.jsonl
```

### Plotting Metrics

```bash
# Visualise reward curves and PPA trends after training
python scripts/plot_metrics.py --log-file logs/hard.jsonl
```

The plotting script generates reward/wirelength/thermal/illegal-rate trend charts across episodes, making it easy to compare reward-weight configurations and model providers side by side.

### LLM Configuration

| Variable | Values | Description |
|----------|--------|-------------|
| `LLM_PROVIDER` | `dummy`, `openai` | Backend selection (`openai` also covers Groq/OpenAI-compatible APIs) |
| `LLM_MODEL` | e.g. `llama-3.1-8b-instant` | Model name |
| `LLM_API_KEY` | `gsk-...` or provider key | Primary API key variable |
| `LLM_BASE_URL` | e.g. `https://api.groq.com/openai/v1/` | OpenAI-compatible endpoint override |
| `OPENAI_API_KEY` | `sk-...` (optional fallback) | Secondary key fallback supported by code |

---

## 10  Real Training Evidence (End-to-End)

This project is beyond "training script exists." The full loop runs on the environment:

1. Reset environment with `SiliconFlowPPAEnvironment.reset()`
2. Generate action (`llm_act(obs)` or dummy baseline)
3. Step environment with legality + PPA metric updates (`env.step(action)`)
4. Compute reward and write JSONL logs (`--log-file`)
5. Generate plots from committed logs (`scripts/plot_metrics.py`)

### Submission-Time Evidence Status

We are submitting verified end-to-end training evidence from committed runs (`logs/*.jsonl` + `plots/*.png`) to meet the deadline.
A longer 50-episode baseline-vs-trained comparison on the same scenario is currently in progress and will be added as an extended benchmark update.

### Measured Results from Committed Logs

The values below are computed from files in `logs/`:

| Run log | Episodes | Avg reward | Reward (first -> last) | Avg illegal moves | Avg wirelength |
|---------|----------|------------|--------------------------|-------------------|----------------|
| `small_run3.jsonl` | 5 | 23.948 | 22.257 -> 23.792 | 0.40 | 2.470 |
| `medium_run2.jsonl` | 5 | 29.151 | 29.520 -> 28.319 | 0.40 | 5.627 |
| `hard_run2.jsonl` | 5 | 39.823 | 39.506 -> 42.910 | 1.00 | 3.882 |

### Plots (Readable + Committed)

The repo already contains readable `.png` outputs in `plots/`:

- `plots/reward_per_episode.png` - episode vs total reward
- `plots/wirelength_per_episode.png` - episode vs final wirelength
- `plots/thermal_per_episode.png` - episode vs thermal max
- `plots/illegal_per_episode.png` - episode vs illegal placements
- `plots/summary.png` - run-level aggregate comparison

### During-Training Evidence (Qualitative)

In addition to logs/plots, we also include during-training artifacts:
- terminal screenshots from Groq-backed training runs
- Google Colab screenshots from SmolLM2 training
- training dataset snapshots/exports used during model development (attached in submission assets)

Screenshot placeholders for submission package:

```md
![Groq terminal training run](assets/groq_training_terminal.png)
![SmolLM2 training in Google Colab](assets/smollm2_colab_training.png)
```

---

## 11  Links to Add Before Submission

Replace placeholders with your real links:

- **GitHub repo:** `https://github.com/<your-username>/<repo-name>`
- **Hugging Face model:** `https://huggingface.co/<your-username>/<model-name>`
- **Hugging Face environment (Space/OpenEnv):** `https://huggingface.co/spaces/<your-username>/<space-name>`
- **Weights & Biases run (optional but recommended):** `https://wandb.ai/<entity>/<project>/runs/<run-id>`

---

## 12  Beyond Proxies — EDA Integration

The proxy metrics used in reward signal computation for SiliconFlow-PPA are analytical metrics that estimate actual PPA without running the EDA process. This is great for research (fast experiments, license independence), but in order to fully evaluate floorplan quality, we must use EDA tool results.

In our repository, you will find EDA evaluation code located at `scripts/run_eda_eval.py` and `src/eda_flow/`. The process is done through the following steps:

1. Export the final floorplan of this episode in the format suitable for OpenROAD flows
2. Do staged EDA evaluations (Synthesis -> Floorplan -> Placement -> Clock Tree Synthesis -> Routing)
3. Extract the actual PPA results from reports generated by EDA tool
4. Compare these results to the PPA metric

This helps to see the discrepancy between proxy and true metrics, which can help researchers tune proxy reward weights later on.

> **Future work:** Generating trajectory datasets from EDA-evaluated episodes opens the door to offline RL and imitation learning — training a policy purely from good placements without the expense of running RL online against real EDA tools.

For judging clarity: current submission demonstrates a complete RL environment + proxy-metric training pipeline, and includes an EDA starter pipeline (`scripts/run_eda_eval.py`, `src/eda_flow/`) that bridges to real OpenROAD report-based evaluation.

---

## 13  Current Status — What Works Today

SiliconFlow-PPA is best understood as a **working RL environment and experimentation harness** for autonomous floorplanning research — not a finished product, but a rigorous platform for building one.

| Feature | Status |
|---------|--------|
|  Reproducible Training | End-to-end loops run cleanly across all three scenarios with deterministic seeds |
|  JSONL Logging | Every episode produces structured step-level logs: actions, rewards, PPA metrics |
|  Metric Visualisation | Plotting utilities show reward curves, wirelength trends, thermal peaks, illegal-placement rates |
|  Multi-Provider | Groq/OpenAI-compatible and dummy agent both work out of the box — provider swap is a one-flag change |
|  Test Suite | Pytest unit and integration tests cover environment mechanics, reward functions, and API endpoints |
|  EDA Scaffold | OpenROAD-style evaluation pipeline in place for proxy-vs-real metric comparison |

---

## 14  Roadmap — What Comes Next

1. **Benchmark sweeps** — Run larger experiments across scenarios and reward-weight configurations to map the Pareto frontier of PPA trade-offs.
2. **Runtime optimisation** — Profile and optimise step-path latency; cache intermediate metric computations and reduce API round-trips.
3. **Stronger baselines** — Add beam search over candidate placements, simulated annealing, and fine-tuned smaller models via TRL + Unsloth.
4. **Real EDA integration** — Connect a full OpenROAD flow and quantify the proxy vs. real PPA gap across 100+ episodes on the hard scenario.
5. **Trajectory datasets** — Build curated datasets from EDA-validated episodes for offline RL and imitation learning experiments.
6. **OpenEnv hub publication** — Publish the environment to the Hugging Face OpenEnv hub so the broader RL community can train and benchmark against it.

---

**Project repository:** [SiliconFlow-PPA on GitHub](https://github.com/SujalDeshmukh/SiliconFlow_PPA)  
