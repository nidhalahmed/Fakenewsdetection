## Politifact

| Model Variant                    | Features Used                  | Accuracy   | F1 (Macro) | Precision | Recall | AUC        |
| -------------------------------- | ------------------------------ | ---------- | ---------- | --------- | ------ | ---------- |
| **GIN Baseline**                 | BERT only                      | **0.8054** | **0.8015** | 0.8344    | 0.7754 | 0.8846     |
| **+ Topology (Xu)**              | + Degree, Clustering           | 0.7647     | 0.7587     | 0.7218    | 0.8773 | 0.8854     |
| **+ Topology + PageRank**        | + Degree, Clustering, PageRank | 0.4932     | 0.3362     | 0.2896    | 0.0078 | 0.8745     |
| **+ Topology + Temporal (GNN+)** | + Topology + Temporal          | **0.8235** | **0.8200** | 0.7962    | 0.8863 | **0.9081** |
| **+ PageRank**                   | + PageRank                     | **0.8190** | **0.8162** | 0.8187    | 0.8308 | **0.9078** |


## Gossipcop

| Model Variant                    | Features Used                  | Accuracy   | F1 (Macro) | Precision  | Recall | AUC        |
| -------------------------------- | ------------------------------ | ---------- | ---------- | ---------- | ------ | ---------- |
| **GIN Baseline**                 | BERT only                      | 0.7360     | 0.7145     | 0.9868     | 0.4799 | 0.9817     |
| **+ Topology (Xu)**              | + Degree, Clustering           | **0.9456** | **0.9450** | 0.9687     | 0.9209 | 0.9828     |
| **+ Topology + PageRank**        | + Degree, Clustering, PageRank | **0.9548** | **0.9542** | 0.9481     | 0.9622 | 0.9839     |
| **+ Topology + Temporal (GNN+)** | + Topology + Temporal          | 0.9543     | 0.9537     | **0.9730** | 0.9341 | **0.9854** |
| **+ PageRank**                   | + PageRank                     | **0.9088** | **0.9072** | 0.9800    | 0.8352 | **0.9790** |