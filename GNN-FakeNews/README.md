
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

* Reproduce a **baseline graph-based fake news detection model (UPFD-style)**
* Compare **content vs propagation-based approaches**
* Evaluate the impact of:

  * Structural (topological) features
  * Temporal propagation features
* Perform **controlled experiments on the same dataset (FakeNewsNet)**

---

## 🗂️ Repository Structure

```
.
├── data/                  # FakeNewsNet dataset (Politifact, GossipCop)
├── gnn/
│   ├── models/            # Model implementations
│   │   ├── dou.py         # UPFD
│   │   ├── xu_baseline.py # Topological feature model
│   │   ├── xu_pagerank.py # Topology + PageRank
│   │   └── temporal.py    # Temporal feature model
│   │
│   └── analysis/          # Feature extraction scripts
│       └── temporal_analysis.py
│
├── results/               # Generated outputs
│   ├── *.csv              # Temporal metrics
│   └── temporal_plots/    # Visualizations
│
├── utils/                 # Data loading & evaluation utilities
├── scripts/               # Run scripts (optional)
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
python gnn/models/dou.py --dataset politifact --feature bert
```

### Topological Features (Xu-style)

```bash
python gnn/models/xu_baseline.py --dataset politifact --feature bert
```

### + PageRank

```bash
python gnn/models/xu_pagerank.py --dataset politifact --feature bert
```

### Temporal Features

```bash
python gnn/models/temporal.py --dataset politifact --feature bert
```

---

## 📊 Feature Engineering

### Structural Features

* Degree centrality
* Clustering coefficient
* PageRank

Computed using **NetworkX** and appended to node features.

### Temporal Features

Extracted from propagation timestamps:

* Cascade size
* Lifetime
* Burstiness
* Inter-arrival statistics

Generated via:

```bash
python gnn/analysis/temporal_analysis.py
```

---

## 📈 Experimental Setup

* Dataset: **FakeNewsNet**

  * Politifact
  * GossipCop
* Task: Graph classification (fake vs real)
* Models evaluated:

  * Text-only baseline (optional)
  * Graph baseline (UPFD-style)
  * Graph + topology
  * Graph + temporal features

---

## 🔍 Key Insight (Preliminary)

* Graph-based models outperform purely text-based approaches in many cases
* Structural and temporal features show **mixed impact**
* Improvements appear **dataset-dependent**, suggesting:

  * Propagation patterns differ across domains
  * GNNs may already capture some structural signals implicitly

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
