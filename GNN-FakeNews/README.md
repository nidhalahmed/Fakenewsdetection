
---

# 🧠 GNN-based Fake News Detection with Structural & Temporal Features

## 📌 Overview

This project explores **fake news detection on social media** using both:

* **Content-based methods** (textual features)
* **Graph-based methods** (propagation structure)

We build upon the **User Preference-aware Fake News Detection (UPFD)** framework and investigate whether adding:

* **Topological features** (e.g., degree, clustering, PageRank)
* **Temporal features** (e.g., cascade dynamics, burstiness)

can improve performance.

---

## 🎯 Objectives

## 🎯 Objectives

* Reproduce a controlled **graph-based fake news detection baseline** on FakeNewsNet
* Evaluate whether adding explicit:
  * **topological features** (degree, clustering, PageRank)
  * **temporal propagation features** (cascade dynamics, burstiness)
  improves graph classification performance
* Compare feature variants under a **fixed graph architecture**
* Analyze how results differ across **Politifact** and **GossipCop**
---

## 🗂️ Repository Structure

```
.
├── data/                  # FakeNewsNet dataset (Politifact, GossipCop)
├── gnn/
│   ├── models/            # Main graph models
│   │   ├── gin_base.py
│   │   ├── gin_topo.py
│   │   ├── gin_topo_pr.py
│   │   ├── gin_topo_temp.py
│   │   ├── gin_temp.py
│   │   └── extra_models/  # Additional exploratory variants
│   │       ├── gin_pr.py
│   │       ├── gin_temp_pr.py
│   │       └── ginplus.py
│   └── analysis/
│       └── temporal_analysis.py
├── results/
│   ├── logs/
│   ├── output.md
│   ├── politifact_temporal_metrics.csv
│   ├── gossipcop_temporal_metrics.csv
│   └── temporal_plots/
├── scripts/
│   └── run_models.sh
├── utils/
│   ├── data_loader.py
│   ├── eval_helper.py
│   ├── profile_feature.py
│   └── twitter_crawler.py
├── README.md
└── requirements.txt
```

---

## ⚙️ Setup

### Requirements

* Python ≥ 3.8
* PyTorch
* PyTorch Geometric
* NetworkX
* NumPy, Pandas, Matplotlib

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🚀 Running Models

Run from the project root:

### Baseline (UPFD-style)

```bash
python gnn/models/gin_base.py --dataset politifact --feature bert
```

### Topological Features (Xu-style)

```bash
python gnn/models/gin_topo.py --dataset politifact --feature bert
```

### + PageRank

```bash
python gnn/models/gin_topo_pr.py --dataset politifact --feature bert
```

### + Temporal Features

```bash
python gnn/models/gin_topo_temp.py --dataset politifact --feature bert
```

### Pure Temporal Features

```bash
python gnn/models/gin_temp.py --dataset politifact --feature bert
```

---

## 📊 Feature Engineering

### Structural Features
Computed with NetworkX and appended to node features:
* Degree centrality
* Clustering coefficient
* PageRank

### Temporal Features
Computed from propagation timestamps at the graph level:
* Cascade size
* Lifetime
* Burstiness
* t50

Additional descriptive temporal statistics are also extracted during analysis and saved as CSVs for inspection.
---

## 📈 Experimental Setup

* Dataset: **FakeNewsNet**
  * Politifact
  * GossipCop
* Task: Graph classification (**fake vs real**)
* Input node features: **BERT-based features**
* Graph architecture: **GIN + attention-based pooling**
* Data split: fixed **UPFD / FakeNewsNet benchmark split**
* Main evaluated variants:
  * `gin_base` — baseline
  * `gin_topo` — + topology
  * `gin_topo_pr` — + topology + PageRank
  * `gin_topo_temp` — + topology + temporal
  * `gin_temp` — + temporal

---

## 🔍 Key Findings

* Feature effectiveness is **dataset-dependent**
* On **Politifact**, temporal features are the strongest addition among the tested variants
* On **GossipCop**, richer structural combinations perform better
* Explicit structural features do not uniformly improve performance, suggesting that:
  * some signals may already be captured implicitly by the GNN
  * additional features can introduce noise, especially in smaller datasets
  
---

## 🙏 Acknowledgment

This project builds upon the open-source **GNN-FakeNews / UPFD** implementation.

* Original repository: safe-graph/GNN-FakeNews
* License: Apache License 2.0
* Paper:

  > Dou, Y., Shu, K., Xia, C., Yu, P. S., & Sun, L. (2021).
  > *User Preference-aware Fake News Detection*

We adapted the codebase and introduced:

* Topological feature augmentation
* Temporal propagation features
* Additional experimental comparisons

---

## 📌 Future Work

* More advanced temporal modeling (e.g., dynamic GNNs)
* Cross-dataset generalization experiments
* Feature importance analysis
* Better fusion of graph-level and node-level signals

---

## 👤 Author

Nidhal Ahmed
B.Tech CSE, NIT Calicut

---
