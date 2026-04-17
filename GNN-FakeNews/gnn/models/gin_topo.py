import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import argparse
from tqdm import tqdm

import torch
import torch.nn.functional as F

from torch_geometric.nn import GINConv, AttentionalAggregation
from torch_geometric.loader import DataLoader, DataListLoader

import networkx as nx
import numpy as np

from utils.data_loader import *
from utils.eval_helper import *

from torch.utils.data import Subset

# =========================
# Topology Features (SAFE VERSION)
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

def safe_zscore(arr):
    std = arr.std()
    if std < 1e-8:
        return np.zeros_like(arr)
    return (arr - arr.mean()) / std

def add_features_to_dataset(dataset):
    print("Computing topology features once...")

    data_list = []
    for i in range(len(dataset)):
        data = dataset.get(i)   # IMPORTANT
        topo = compute_topology_features(data)
        data.x = torch.cat([data.x, topo], dim=1)
        data_list.append(data)

    dataset.data, dataset.slices = dataset.collate(data_list)
    return dataset


# =========================
# Model
# =========================
class Model(torch.nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()

        self.num_features = args.num_features
        self.nhid = args.nhid
        self.num_classes = args.num_classes
        self.dropout_ratio = args.dropout_ratio

        nn1 = torch.nn.Sequential(
            torch.nn.Linear(self.num_features, self.nhid),
            torch.nn.ReLU(),
            torch.nn.Linear(self.nhid, self.nhid)
        )
        self.conv1 = GINConv(nn1)

        self.att_pool = AttentionalAggregation(
            gate_nn=torch.nn.Sequential(
                torch.nn.Linear(self.nhid, 1)
            )
        )

        self.bn = torch.nn.BatchNorm1d(self.nhid)

        self.lin2 = torch.nn.Linear(self.nhid, self.num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.relu(self.conv1(x, edge_index))
        pooled = self.att_pool(x, batch)

        pooled = self.bn(pooled)
        pooled = F.dropout(pooled, p=self.dropout_ratio, training=self.training)

        out = F.log_softmax(self.lin2(pooled), dim=-1)
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


dataset = add_features_to_dataset(dataset)

args.num_features = dataset[0].x.size(1)
print("num_features =", args.num_features)

args.num_classes = dataset.num_classes
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
model = Model(args)

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

    [acc, f1_macro, f1_micro, precision, recall, auc, ap], _ = compute_test(test_loader)

    print(
        f'Test set results: acc: {acc:.4f}, f1_macro: {f1_macro:.4f}, '
        f'f1_micro: {f1_micro:.4f}, precision: {precision:.4f}, '
        f'recall: {recall:.4f}, auc: {auc:.4f}, ap: {ap:.4f}'
    )