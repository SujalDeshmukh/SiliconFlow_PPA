"""
SiliconFlow-PPA: Plot training metrics from JSONL logs.
Generates reward, wirelength, and thermal plots for all log files.
"""
import json
import os
import glob
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ---------------------------------------------------------------------------
# Load all log files
# ---------------------------------------------------------------------------

LOG_DIR = "logs"
OUTPUT_DIR = "plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

log_files = glob.glob(os.path.join(LOG_DIR, "*.jsonl"))

if not log_files:
    print("No log files found in logs/")
    exit(1)

print(f"Found {len(log_files)} log files:")
for f in log_files:
    print(f"  {f}")

# ---------------------------------------------------------------------------
# Parse logs
# ---------------------------------------------------------------------------

all_runs = {}

for log_file in log_files:
    name = os.path.basename(log_file).replace(".jsonl", "")
    episodes, rewards, wirelen, thermal, illegal = [], [], [], [], []

    with open(log_file) as f:
        for line in f:
            d = json.loads(line)
            episodes.append(d["episode"])
            rewards.append(d["total_reward"])
            wirelen.append(d["wirelength"])
            thermal.append(d["thermal_max"])
            illegal.append(d.get("illegal", 0))

    all_runs[name] = {
        "episodes": episodes,
        "rewards": rewards,
        "wirelen": wirelen,
        "thermal": thermal,
        "illegal": illegal,
    }

# ---------------------------------------------------------------------------
# Plot 1: Total Reward per Episode (all runs)
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(10, 5))
colors = plt.cm.tab10.colors

for i, (name, data) in enumerate(all_runs.items()):
    ax.plot(data["episodes"], data["rewards"],
            marker="o", linewidth=2, color=colors[i % 10], label=name)

ax.set_xlabel("Episode")
ax.set_ylabel("Total Reward")
ax.set_title("SiliconFlow-PPA: Total Reward per Episode")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "reward_per_episode.png"), dpi=150)
print(f"Saved: {OUTPUT_DIR}/reward_per_episode.png")
plt.close()

# ---------------------------------------------------------------------------
# Plot 2: Wirelength per Episode
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(10, 5))

for i, (name, data) in enumerate(all_runs.items()):
    ax.plot(data["episodes"], data["wirelen"],
            marker="s", linewidth=2, color=colors[i % 10], label=name)

ax.set_xlabel("Episode")
ax.set_ylabel("Final Wirelength")
ax.set_title("SiliconFlow-PPA: Wirelength per Episode (lower is better)")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "wirelength_per_episode.png"), dpi=150)
print(f"Saved: {OUTPUT_DIR}/wirelength_per_episode.png")
plt.close()

# ---------------------------------------------------------------------------
# Plot 3: Thermal Max per Episode
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(10, 5))

for i, (name, data) in enumerate(all_runs.items()):
    ax.plot(data["episodes"], data["thermal"],
            marker="^", linewidth=2, color=colors[i % 10], label=name)

ax.axhline(y=1.0, color="red", linestyle="--", alpha=0.7, label="Thermal limit (1.0)")
ax.set_xlabel("Episode")
ax.set_ylabel("Thermal Max")
ax.set_title("SiliconFlow-PPA: Thermal Max per Episode (lower is better)")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "thermal_per_episode.png"), dpi=150)
print(f"Saved: {OUTPUT_DIR}/thermal_per_episode.png")
plt.close()

# ---------------------------------------------------------------------------
# Plot 4: Illegal Moves per Episode
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(10, 5))

for i, (name, data) in enumerate(all_runs.items()):
    ax.plot(data["episodes"], data["illegal"],
            marker="x", linewidth=2, color=colors[i % 10], label=name)

ax.set_xlabel("Episode")
ax.set_ylabel("Illegal Placements")
ax.set_title("SiliconFlow-PPA: Illegal Moves per Episode (lower is better)")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "illegal_per_episode.png"), dpi=150)
print(f"Saved: {OUTPUT_DIR}/illegal_per_episode.png")
plt.close()

# ---------------------------------------------------------------------------
# Plot 5: Summary bar chart
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(14, 5))

names = list(all_runs.keys())
avg_rewards = [sum(d["rewards"]) / len(d["rewards"]) for d in all_runs.values()]
avg_wirelen  = [sum(d["wirelen"]) / len(d["wirelen"]) for d in all_runs.values()]
avg_thermal  = [sum(d["thermal"]) / len(d["thermal"]) for d in all_runs.values()]

x = range(len(names))

axes[0].bar(x, avg_rewards, color=[colors[i % 10] for i in range(len(names))])
axes[0].set_xticks(x)
axes[0].set_xticklabels(names, rotation=15, ha="right", fontsize=8)
axes[0].set_ylabel("Avg Total Reward")
axes[0].set_title("Avg Reward per Run")
axes[0].grid(True, alpha=0.3, axis="y")

axes[1].bar(x, avg_wirelen, color=[colors[i % 10] for i in range(len(names))])
axes[1].set_xticks(x)
axes[1].set_xticklabels(names, rotation=15, ha="right", fontsize=8)
axes[1].set_ylabel("Avg Wirelength")
axes[1].set_title("Avg Wirelength per Run")
axes[1].grid(True, alpha=0.3, axis="y")

axes[2].bar(x, avg_thermal, color=[colors[i % 10] for i in range(len(names))])
axes[2].axhline(y=1.0, color="red", linestyle="--", alpha=0.7, label="Limit")
axes[2].set_xticks(x)
axes[2].set_xticklabels(names, rotation=15, ha="right", fontsize=8)
axes[2].set_ylabel("Avg Thermal Max")
axes[2].set_title("Avg Thermal per Run")
axes[2].legend()
axes[2].grid(True, alpha=0.3, axis="y")

plt.suptitle("SiliconFlow-PPA Training Summary", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "summary.png"), dpi=150)
print(f"Saved: {OUTPUT_DIR}/summary.png")
plt.close()

print(f"\n✅ All plots saved to {OUTPUT_DIR}/")
print("\nSummary:")
for name, data in all_runs.items():
    avg_r = sum(data["rewards"]) / len(data["rewards"])
    avg_w = sum(data["wirelen"]) / len(data["wirelen"])
    print(f"  {name}: avg_reward={avg_r:.2f}  avg_wl={avg_w:.2f}")