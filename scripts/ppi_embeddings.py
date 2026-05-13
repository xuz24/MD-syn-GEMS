import torch
import torch.nn as nn
import pandas as pd
import networkx as nx
from node2vec import Node2Vec

ppi_path = "/home/xuzijie/MD-syn-GEMS/data/string_ppi_entrez.csv"
landmarks_path = "/home/xuzijie/MD-syn-GEMS/data/landmark_genes.txt"

print(f'=== PPI FILE: {ppi_path} ===')
print(f'=== LANDMARKS FILE: {landmarks_path} ===')

ppi_df = pd.read_csv(ppi_path)
landmarks_df = pd.read_csv(landmarks_path, sep="\t")

landmark_ids = set(list(landmarks_df["Entrez ID"]))

ppi_filtered = ppi_df[
    ppi_df["entrez_id_a"].isin(landmark_ids) &
    ppi_df["entrez_id_b"].isin(landmark_ids)
]

print('=== START PROCESSING ===')

G = nx.from_pandas_edgelist(ppi_filtered, source="entrez_id_a", target="entrez_id_b",
edge_attr="score")
node2vec = Node2Vec(G, dimensions=128, walk_length=80, num_walks=10, workers=4, weight_key="score")
model = node2vec.fit(window=10, min_count=1, batch_words=4)

nodes = model.wv.index_to_key  # list of node IDs (strings)

node_to_idx = {n: i for i, n in enumerate(nodes)}

emb_matrix = torch.stack([
    torch.tensor(model.wv[n]) for n in nodes
])

torch.save({
    "embeddings": emb_matrix,
    "node_to_idx": node_to_idx,
    "nodes": nodes
}, "ppi_node2vec_matrix.pt")