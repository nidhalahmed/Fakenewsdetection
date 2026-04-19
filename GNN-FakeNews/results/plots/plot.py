import matplotlib.pyplot as plt
import numpy as np

models = ["Stack", "BERT", "GIN (Base)", "+Temp", "+Topo", "+Topo+Temp", "+Topo+PR"]

f1_pf = [0.8210, 0.8839, 0.8015, 0.7587, 0.8584, 0.3362, 0.7665]
f1_gc = [0.8978, 0.9130, 0.7145, 0.9450, 0.9270, 0.9542, 0.9502]

graph_models = ["GIN", "+Topo", "+Temp", "+Topo+PR", "+Topo+Temp"]
pf_graph = [0.8015, 0.7587, 0.8584, 0.3362, 0.7665]
gc_graph = [0.7145, 0.9450, 0.9270, 0.9542, 0.9502]


plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150
})

def add_value_labels(ax, bars, fmt="{:.3f}"):
    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + 0.005,
            bar.get_y() + bar.get_height() / 2,
            fmt.format(width),
            va="center",
            ha="left"
        )

fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True)

datasets = [
    ("Politifact", f1_pf),
    ("GossipCop", f1_gc)
]

for ax, (title, scores) in zip(axes, datasets):
    y = np.arange(len(models))

    best_idx = int(np.argmax(scores))
    colors = [] 

    for i, m in enumerate(models):
        if i < 2:
            colors.append("#4c78a8")  # text models (green)
        else:
            colors.append("#c7d4e8")  # graph models (blue-gray)
    colors[best_idx] = "#59a14f"

    bars = ax.barh(y, scores, color=colors, edgecolor="black", linewidth=0.6)

    ax.set_yticks(y)
    ax.set_yticklabels(models)
    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_xlabel("Macro F1")
    ax.set_xlim(0, 1.02)
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    add_value_labels(ax, bars)

fig.suptitle("Model Performance Comparison Across Datasets", fontsize=15, y=1.02)
plt.tight_layout()
plt.savefig("model_comparison_report.png", bbox_inches="tight")
plt.close()

x = np.arange(len(graph_models))
width = 0.36

fig, ax = plt.subplots(figsize=(10, 5.5))

bars1 = ax.bar(x - width/2, pf_graph, width, label="Politifact",
               color="#4c78a8", edgecolor="black", linewidth=0.6)
bars2 = ax.bar(x + width/2, gc_graph, width, label="GossipCop",
               color="#f28e2b", edgecolor="black", linewidth=0.6)

ax.set_title("Impact of Structural and Temporal Features on Graph Models")
ax.set_ylabel("Macro F1")
ax.set_xticks(x)
ax.set_xticklabels(graph_models)
ax.set_ylim(0, 1.02)
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.set_axisbelow(True)
ax.legend(frameon=False)

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2,
            height + 0.015,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=9
        )

plt.tight_layout()
plt.savefig("graph_feature_impact_report.png", bbox_inches="tight")
plt.close()