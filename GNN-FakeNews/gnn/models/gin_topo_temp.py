import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import argparse
import pickle
from pathlib import Path

from tqdm import tqdm

import torch
import torch.nn.functional as F

from torch_geometric.nn import GINConv, AttentionalAggregation
from torch_geometric.loader import DataLoader, DataListLoader

import networkx as nx
import numpy as np
import pandas as pd

from utils.data_loader import *
from utils.eval_helper import *

from torch.utils.data import Subset


# =========================
# Paths
# =========================
BASE_DIR = Path(__file__).resolve().parents[2]
RESULTS_DIR = BASE_DIR / "results"


# =========================
# Helpers
# =========================
def safe_zscore(arr):
    arr = np.asarray(arr, dtype=np.float32)
    std = arr.std()
    if std < 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - arr.mean()) / std


# =========================
# Node-level topology features
# =========================
def compute_topology_features(data):
    edge_index = data.edge_index.cpu()
    num_nodes = data.num_nodes

    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    G.add_edges_from(edge_index.t().tolist())

    degree = np.array(list(nx.degree_centrality(G).values()), dtype=np.float32)
    clustering = np.array(list(nx.clustering(G).values()), dtype=np.float32)

    degree = safe_zscore(degree)
    clustering = safe_zscore(clustering)

    topo = torch.tensor(
        np.stack([degree, clustering], axis=1),
        dtype=torch.float
    )
    return topo


def add_topology_features_to_dataset(dataset):
    print("Computing topology features once...")

    data_list = []
    for i in range(len(dataset)):
        data = dataset.get(i)
        topo = compute_topology_features(data)
        data.x = torch.cat([data.x, topo], dim=1)
        data_list.append(data)

    dataset.data, dataset.slices = dataset.collate(data_list)
    return dataset


# =========================
# Graph-level temporal features
# =========================
def load_temporal_feature_table(dataset_name):
    """
    Expects CSV created by project/temporal_analysis.py
    Must contain graph_id and the chosen temporal columns.
    """
    csv_path = RESULTS_DIR / f"{dataset_name}_temporal_metrics.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Temporal feature CSV not found: {csv_path}\n"
            f"Expected file in results/. Generate it first if missing."
        )

    df = pd.read_csv(csv_path)

    required_cols = ["graph_id", "cascade_size", "lifetime", "burstiness", "t50"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {csv_path}: {missing}")

    df = df.sort_values("graph_id").reset_index(drop=True)

    # Log-transform highly skewed temporal features
    for col in ["cascade_size", "lifetime", "t50"]:
        df[col] = np.log1p(df[col].astype(np.float32))

    # Keep burstiness as-is
    feature_cols = ["cascade_size", "lifetime", "burstiness", "t50"]
    feat = df[feature_cols].to_numpy(dtype=np.float32)

    return feat, feature_cols


def normalize_graph_features_train_only(feat_matrix, train_idx):
    """
    Train-split normalization to avoid leakage.
    """
    feat_matrix = np.asarray(feat_matrix, dtype=np.float32)
    train_feat = feat_matrix[train_idx]

    mean = train_feat.mean(axis=0, keepdims=True)
    std = train_feat.std(axis=0, keepdims=True)
    std[std < 1e-8] = 1.0

    feat_norm = (feat_matrix - mean) / std
    return feat_norm.astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


def add_graph_features_to_dataset(dataset, dataset_name):
    print("Loading graph-level temporal features...")

    graph_feat, feature_cols = load_temporal_feature_table(dataset_name)

    if len(graph_feat) != len(dataset):
        raise ValueError(
            f"Temporal feature rows ({len(graph_feat)}) != dataset size ({len(dataset)})"
        )

    train_idx = dataset.train_idx.tolist()
    graph_feat, mean, std = normalize_graph_features_train_only(graph_feat, train_idx)

    print("Graph feature columns:", feature_cols)
    print("Graph feature mean (train):", mean.flatten())
    print("Graph feature std  (train):", std.flatten())

    data_list = []
    for i in range(len(dataset)):
        data = dataset.get(i)
        data.graph_feat = torch.tensor(graph_feat[i], dtype=torch.float).unsqueeze(0)
        data_list.append(data)

    dataset.data, dataset.slices = dataset.collate(data_list)
    return dataset, len(feature_cols)


# =========================
# Model
# =========================
class Model(torch.nn.Module):
    def __init__(self, args, concat=False):
        super(Model, self).__init__()

        self.num_features = args.num_features
        self.num_graph_features = args.num_graph_features
        self.nhid = args.nhid
        self.num_classes = args.num_classes
        self.dropout_ratio = args.dropout_ratio
        self.concat = concat

        # GIN encoder
        nn1 = torch.nn.Sequential(
            torch.nn.Linear(self.num_features, self.nhid),
            torch.nn.ReLU(),
            torch.nn.Linear(self.nhid, self.nhid)
        )
        self.conv1 = GINConv(nn1)

        # Attention pooling
        self.att_pool = AttentionalAggregation(
            gate_nn=torch.nn.Sequential(
                torch.nn.Linear(self.nhid, 1)
            )
        )

        self.bn_graph = torch.nn.BatchNorm1d(self.nhid)

        # Optional root-node fusion, kept from your original structure
        if self.concat:
            self.lin0 = torch.nn.Linear(self.num_features, self.nhid)
            self.lin1 = torch.nn.Linear(self.nhid * 2, self.nhid)

        # Small MLP for graph-level temporal features
        self.graph_feat_mlp = torch.nn.Sequential(
            torch.nn.Linear(self.num_graph_features, self.nhid // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(self.dropout_ratio)
        )

        # Final fusion
        self.fusion_bn = torch.nn.BatchNorm1d(self.nhid + self.nhid // 2)
        self.lin_out = torch.nn.Linear(self.nhid + self.nhid // 2, self.num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        graph_feat = data.graph_feat

        # Node encoder
        x = F.relu(self.conv1(x, edge_index))

        # Graph pooling
        pooled = self.att_pool(x, batch)
        pooled = self.bn_graph(pooled)
        pooled = F.dropout(pooled, p=self.dropout_ratio, training=self.training)

        # Graph-level temporal branch
        if graph_feat.dim() == 1:
            graph_feat = graph_feat.view(1, -1)
        elif graph_feat.dim() > 2:
            graph_feat = graph_feat.view(graph_feat.size(0), -1)

        graph_feat = self.graph_feat_mlp(graph_feat)

        # Fuse pooled GNN embedding + temporal graph features
        fused = torch.cat([pooled, graph_feat], dim=1)
        fused = self.fusion_bn(fused)
        fused = F.dropout(fused, p=self.dropout_ratio, training=self.training)

        out = F.log_softmax(self.lin_out(fused), dim=-1)
        return out


# =========================
# Evaluation
# =========================
@torch.no_grad()
def compute_test(loader):
    model.eval()
    loss_test = 0.0
    out_log = []

    for data in loader:
        if not args.multi_gpu:
            data = data.to(args.device)

        out = model(data)

        if args.multi_gpu:
            y = torch.cat([d.y.unsqueeze(0) for d in data]).squeeze().to(out.device)
        else:
            y = data.y

        out_log.append([F.softmax(out, dim=1), y])
        loss_test += F.nll_loss(out, y).item()

    return eval_deep(out_log, loader), loss_test


# =========================
# Args
# =========================
parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=777)
parser.add_argument('--device', type=str, default='cpu')

parser.add_argument('--dataset', type=str, default='politifact')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--nhid', type=int, default=128)
parser.add_argument('--dropout_ratio', type=float, default=0.5)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--concat', type=bool, default=False)
parser.add_argument('--multi_gpu', type=bool, default=False)
parser.add_argument('--feature', type=str, default='bert')

args = parser.parse_args()

torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)


# =========================
# Dataset
# =========================
dataset = FNNDataset(
    root='data',
    feature=args.feature,
    empty=False,
    name=args.dataset,
    transform=ToUndirected()
)

# 1) add node-level topology features
dataset = add_topology_features_to_dataset(dataset)

# 2) add graph-level temporal features
dataset, args.num_graph_features = add_graph_features_to_dataset(dataset, args.dataset)

args.num_features = dataset[0].x.size(1)
args.num_classes = dataset.num_classes

print("num_features =", args.num_features)
print("num_graph_features =", args.num_graph_features)
print("num_classes =", args.num_classes)
print(args)

loader_cls = DataListLoader if args.multi_gpu else DataLoader

train_dataset = Subset(dataset, dataset.train_idx.tolist())
val_dataset = Subset(dataset, dataset.val_idx.tolist())
test_dataset = Subset(dataset, dataset.test_idx.tolist())

train_loader = loader_cls(train_dataset, batch_size=args.batch_size, shuffle=True)
val_loader = loader_cls(val_dataset, batch_size=args.batch_size, shuffle=False)
test_loader = loader_cls(test_dataset, batch_size=args.batch_size, shuffle=False)


# =========================
# Model
# =========================
model = Model(args, concat=args.concat)

if args.multi_gpu:
    from torch_geometric.nn import DataParallel
    model = DataParallel(model)

model = model.to(args.device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=args.lr,
    weight_decay=args.weight_decay
)


# =========================
# Training
# =========================
if __name__ == '__main__':

    best_val_f1 = -1.0
    best_state = None
    patience = 10
    patience_counter = 0

    for epoch in tqdm(range(args.epochs)):
        model.train()
        loss_train = 0.0
        out_log = []

        for data in train_loader:
            optimizer.zero_grad()

            if not args.multi_gpu:
                data = data.to(args.device)

            out = model(data)

            if args.multi_gpu:
                y = torch.cat([d.y.unsqueeze(0) for d in data]).squeeze().to(out.device)
            else:
                y = data.y

            loss = F.nll_loss(out, y)
            loss.backward()
            optimizer.step()

            loss_train += loss.item()
            out_log.append([F.softmax(out, dim=1), y])

        acc_train, _, _, _, recall_train, auc_train, _ = eval_deep(out_log, train_loader)
        [acc_val, f1_macro_val, _, _, recall_val, auc_val, _], loss_val = compute_test(val_loader)

        print(
            f'loss_train: {loss_train:.4f}, acc_train: {acc_train:.4f}, '
            f'recall_train: {recall_train:.4f}, auc_train: {auc_train:.4f}, '
            f'loss_val: {loss_val:.4f}, acc_val: {acc_val:.4f}, '
            f'f1_macro_val: {f1_macro_val:.4f}, '
            f'recall_val: {recall_val:.4f}, auc_val: {auc_val:.4f}'
        )

        if f1_macro_val > best_val_f1 + 1e-4:
            best_val_f1 = f1_macro_val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model = model.to(args.device)

    [acc, f1_macro, f1_micro, precision, recall, auc, ap], _ = compute_test(test_loader)

    print(
        f'Test set results: acc: {acc:.4f}, f1_macro: {f1_macro:.4f}, '
        f'f1_micro: {f1_micro:.4f}, precision: {precision:.4f}, '
        f'recall: {recall:.4f}, auc: {auc:.4f}, ap: {ap:.4f}'
    )