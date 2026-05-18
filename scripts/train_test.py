import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from models import MD_Syn

cell_line_embeddings_path = "/home/xuzijie/MD-syn-GEMS/data/CCLE_expression.csv"
cell_line_mapping_path = "/home/xuzijie/MD-syn-GEMS/data/cell_line_ccle_map.csv"
landmarks_path = "/home/xuzijie/MD-syn-GEMS/data/landmark_genes.txt"
molformer_embeddings_path = "/home/xuzijie/MD-syn-GEMS/data/molformer_embeddings_drugcomb.pt"
ppi_graph_path = "/home/xuzijie/MD-syn-GEMS/data/ppi_node2vec_matrix.pt"
drug_targets_path = ["/home/xuzijie/MD-syn-GEMS/data/drug_target.csv", "/home/xuzijie/MD-syn-GEMS/data/targets.csv", "/home/xuzijie/MD-syn-GEMS/data/uniprot_to_entrez.csv"]
drug_dbid_mapping_path = "/home/xuzijie/MD-syn-GEMS/data/drugcomb_name_to_dbid_mapping_improved.csv"
drug_smiles_path = "/home/xuzijie/MD-syn-GEMS/data/molecules.csv"
use_drugcomb = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = MD_Syn(
            cell_line_embeddings_path,
            cell_line_mapping_path,
            landmarks_path,
            molformer_embeddings_path,
            ppi_graph_path,
            drug_targets_path,
            drug_dbid_mapping_path,
            drug_smiles_path,
            use_drugcomb
        ).to(device)

samples = [
    ('DB00063','DB00176','KBM-7',0,1),
    ('DB00063','DB00198','KBM-7',0,1),
    ('DB00063','DB00207','KBM-7',0,1)
    ]

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=5e-4)
model.train()

for drug1, drug2, cell_line, label, fold in samples:
    preds = model(drug1, drug2, cell_line)
    
    label = torch.tensor([label], dtype=torch.long, device=device)
    
    loss = criterion(preds, label)
    
    print(preds, label, loss.item())
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()