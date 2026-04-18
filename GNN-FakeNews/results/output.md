## Politifact

| Model Variant                    | Features Used                  | Accuracy   | F1 (Macro) | Precision  | Recall     | AUC        |
| -------------------------------- | ------------------------------ | ---------- | ---------- | ---------- | ---------- | ---------- |
| **GIN Baseline**                 | BERT only                      | 0.8054     | 0.8015     | 0.8344     | 0.7754     | 0.8846     |
| **+ Topology (Xu)**              | + Degree, Clustering           | 0.7647     | 0.7587     | 0.7218     | 0.8773     | 0.8854     |
| **+ Temporal**                   | + Temporal                     | **0.8597** | **0.8584** | **0.9189** | 0.7977     | 0.9013     |
| **+ Topology + PageRank**        | + Degree, Clustering, PageRank | 0.4932     | 0.3362     | 0.2896     | 0.0078     | 0.8745     |
| **+ Topology + Temporal (GNN+)** | + Topology + Temporal          | 0.7738     | 0.7665     | 0.7313     | **0.8851** | 0.8573     |

## Gossipcop

| Model Variant                    | Features Used                  | Accuracy   | F1 (Macro) | Precision  | Recall     | AUC        |
| -------------------------------- | ------------------------------ | ---------- | ---------- | ---------- | ---------- | ---------- |
| **GIN Baseline**                 | BERT only                      | 0.7360     | 0.7145     | **0.9868** | 0.4799     | 0.9817     |
| **+ Topology (Xu)**              | + Degree, Clustering           | 0.9456     | 0.9450     | 0.9687     | 0.9209     | 0.9828     |
| **+ Temporal**                   | + Temporal                     | 0.9279     | 0.9270     | 0.9499     | 0.9038     | 0.9773     |
| **+ Topology + PageRank**        | + Degree, Clustering, PageRank | **0.9548** | **0.9542** | 0.9481     | **0.9622** | 0.9839     |
| **+ Topology + Temporal (GNN+)** | + Topology + Temporal          | 0.9509     | 0.9502     | 0.9683     | 0.9318     | **0.9873** |