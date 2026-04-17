import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"

DATASET_CONFIG = {
    "politifact": {
        "time_map": DATA_DIR / "pol_id_time_mapping.pkl",
        "twitter_map": DATA_DIR / "pol_id_twitter_mapping.pkl",
        "node_graph_id": DATA_DIR / "politifact" / "raw" / "node_graph_id.npy",
        "graph_labels": DATA_DIR / "politifact" / "raw" / "graph_labels.npy",
    },
    "gossipcop": {
        "time_map": DATA_DIR / "gos_id_time_mapping.pkl",
        "twitter_map": DATA_DIR / "gos_id_twitter_mapping.pkl",
        "node_graph_id": DATA_DIR / "gossipcop" / "raw" / "node_graph_id.npy",
        "graph_labels": DATA_DIR / "gossipcop" / "raw" / "graph_labels.npy",
    },
}


def inspect_pickle(path, name):
    with open(path, "rb") as f:
        obj = pickle.load(f)

    print(f"\n{name}:")
    print("type:", type(obj))

    if isinstance(obj, dict):
        print("num keys:", len(obj))
        sample_items = list(obj.items())[:5]
        print("sample items:", sample_items)
    else:
        print("value preview:", str(obj)[:500])

    return obj


def normalize_labels(graph_labels):
    labels = np.asarray(graph_labels)
    unique = np.unique(labels)

    if set(unique.tolist()) == {0, 1}:
        return labels.astype(int)

    mapping = {val: idx for idx, val in enumerate(sorted(unique.tolist()))}
    return np.array([mapping[x] for x in labels], dtype=int)


def clean_timestamp(ts):
    if ts == "" or ts is None:
        return None
    try:
        return float(ts)
    except (TypeError, ValueError):
        return None


def compute_temporal_metrics(times):
    times = [clean_timestamp(t) for t in times]
    times = [t for t in times if t is not None]

    if len(times) == 0:
        return {
            "cascade_size": 0,
            "num_valid_times": 0,
            "lifetime": np.nan,
            "mean_interarrival": np.nan,
            "median_interarrival": np.nan,
            "std_interarrival": np.nan,
            "burstiness": np.nan,
            "t25": np.nan,
            "t50": np.nan,
            "t75": np.nan,
        }

    times = np.array(sorted(times), dtype=np.float64)

    if len(times) == 1:
        return {
            "cascade_size": 1,
            "num_valid_times": 1,
            "lifetime": 0.0,
            "mean_interarrival": np.nan,
            "median_interarrival": np.nan,
            "std_interarrival": np.nan,
            "burstiness": np.nan,
            "t25": 0.0,
            "t50": 0.0,
            "t75": 0.0,
        }

    deltas = np.diff(times)
    mu = deltas.mean()
    sigma = deltas.std()

    if mu + sigma == 0:
        burstiness = np.nan
    else:
        burstiness = (sigma - mu) / (sigma + mu)

    n = len(times)
    t0 = times[0]

    idx25 = max(0, int(np.ceil(0.25 * n)) - 1)
    idx50 = max(0, int(np.ceil(0.50 * n)) - 1)
    idx75 = max(0, int(np.ceil(0.75 * n)) - 1)

    return {
        "cascade_size": n,
        "num_valid_times": n,
        "lifetime": times[-1] - t0,
        "mean_interarrival": mu,
        "median_interarrival": np.median(deltas),
        "std_interarrival": sigma,
        "burstiness": burstiness,
        "t25": times[idx25] - t0,
        "t50": times[idx50] - t0,
        "t75": times[idx75] - t0,
    }


def build_graph_time_rows(dataset_name, node_graph_id, graph_labels, time_map):
    rows = []

    graph_labels = normalize_labels(graph_labels)
    num_graphs = len(graph_labels)

    for graph_id in range(num_graphs):
        node_ids = np.where(node_graph_id == graph_id)[0]

        times = [time_map.get(int(node_id), None) for node_id in node_ids]
        metrics = compute_temporal_metrics(times)

        row = {
            "dataset": dataset_name,
            "graph_id": graph_id,
            "label": int(graph_labels[graph_id]),
            "num_nodes_in_graph": int(len(node_ids)),
        }
        row.update(metrics)
        rows.append(row)

    return rows


def save_basic_plots(df, dataset_name):
    out_dir = BASE_DIR / "project" / "temporal_plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    label_names = {0: "real", 1: "fake"}
    df_plot = df.copy()
    df_plot["label_name"] = df_plot["label"].map(label_names)

    # Burstiness boxplot
    plt.figure(figsize=(6, 4))
    data = [
        df_plot[df_plot["label"] == 0]["burstiness"].dropna(),
        df_plot[df_plot["label"] == 1]["burstiness"].dropna(),
    ]
    plt.boxplot(data, labels=["real", "fake"])
    plt.ylabel("Burstiness")
    plt.title(f"{dataset_name}: Burstiness by Label")
    plt.tight_layout()
    plt.savefig(out_dir / f"{dataset_name}_burstiness_boxplot.png", dpi=200)
    plt.close()

    # Lifetime boxplot
    plt.figure(figsize=(6, 4))
    data = [
        df_plot[df_plot["label"] == 0]["lifetime"].dropna(),
        df_plot[df_plot["label"] == 1]["lifetime"].dropna(),
    ]
    plt.boxplot(data, labels=["real", "fake"])
    plt.ylabel("Lifetime")
    plt.title(f"{dataset_name}: Lifetime by Label")
    plt.tight_layout()
    plt.savefig(out_dir / f"{dataset_name}_lifetime_boxplot.png", dpi=200)
    plt.close()

    # Cascade size boxplot
    plt.figure(figsize=(6, 4))
    data = [
        df_plot[df_plot["label"] == 0]["cascade_size"].dropna(),
        df_plot[df_plot["label"] == 1]["cascade_size"].dropna(),
    ]
    plt.boxplot(data, labels=["real", "fake"])
    plt.ylabel("Cascade Size")
    plt.title(f"{dataset_name}: Cascade Size by Label")
    plt.tight_layout()
    plt.savefig(out_dir / f"{dataset_name}_cascade_size_boxplot.png", dpi=200)
    plt.close()


def main():
    for dataset in ["politifact", "gossipcop"]:
        print(f"\n=== {dataset.upper()} ===")

        cfg = DATASET_CONFIG[dataset]

        time_map = inspect_pickle(cfg["time_map"], f"{dataset} time_map")
        twitter_map = inspect_pickle(cfg["twitter_map"], f"{dataset} twitter_map")

        node_graph_id = np.load(cfg["node_graph_id"])
        graph_labels = np.load(cfg["graph_labels"])

        print("node_graph_id shape:", node_graph_id.shape)
        print("graph_labels shape:", graph_labels.shape)

        rows = build_graph_time_rows(
            dataset_name=dataset,
            node_graph_id=node_graph_id,
            graph_labels=graph_labels,
            time_map=time_map,
        )

        df = pd.DataFrame(rows)

        print("\nFirst 5 rows:")
        print(df.head())

        print("\nLabel counts:")
        print(df["label"].value_counts(dropna=False).sort_index())

        cols = ["cascade_size", "lifetime", "mean_interarrival", "burstiness", "t50"]

        print("\nGroup means:")
        print(df.groupby("label")[cols].mean(numeric_only=True))

        print("\nGroup medians:")
        print(df.groupby("label")[cols].median(numeric_only=True))

        print("\nMissingness summary:")
        print(df[cols + ["num_valid_times"]].isna().sum())

        out_csv = BASE_DIR / "project" / f"{dataset}_temporal_metrics.csv"
        df.to_csv(out_csv, index=False)
        print(f"\nSaved CSV: {out_csv}")

        save_basic_plots(df, dataset)
        print(f"Saved plots for {dataset}")


if __name__ == "__main__":
    main()