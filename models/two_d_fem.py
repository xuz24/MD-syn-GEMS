from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
import pandas as pd
import networkx as nx
from node2vec import Node2Vec
from rdkit import Chem
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool
import logging


def normalize_drug_name(name: str) -> str:
    # Canonicalize names like "5-FU", "5 FU", and "5fu" to the same key.
    return ''.join(ch for ch in str(name).strip().lower() if ch.isalnum())


def one_hot(x, allowable_set):
    return [int(x == s) for s in allowable_set]


def get_atom_features(atom):
    return (
        one_hot(atom.GetSymbol(), 
                ['C','N','O','F','P','S','Cl','Br','I','H','Unknown']) +
        one_hot(atom.GetDegree(), [0,1,2,3,4,5]) +
        one_hot(atom.GetFormalCharge(), [-1,0,1]) +
        one_hot(atom.GetHybridization(), [
            Chem.rdchem.HybridizationType.SP,
            Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3
        ]) +
        [int(atom.GetIsAromatic())] +
        one_hot(atom.GetTotalNumHs(), [0,1,2,3,4])
    )


def smiles_to_graph(smiles: str) -> Data:
    mol = Chem.MolFromSmiles(smiles)
    # Atom features: atomic number, degree, hybridization, aromaticity, etc.
    atom_features = [get_atom_features(atom) for atom in mol.GetAtoms()]
    x = torch.tensor(atom_features, dtype=torch.float)
    edge_index = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_index += [[i, j], [j, i]] # undirected
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    data = Data(x=x, edge_index=edge_index)
    # For single-graph inputs set batch vector so pooling works
    data.batch = torch.zeros(x.size(0), dtype=torch.long)
    return data


class MolecularGCN(nn.Module):
    def __init__(self, input_dim=29, hidden_dims=[512, 128]):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dims[0])
        self.conv2 = GCNConv(hidden_dims[0], hidden_dims[1])
        self.relu = nn.ReLU()
        
    def forward(self, data):
        # Move data to device if needed
        device = next(self.parameters()).device
        x, edge_index, batch = data.x.to(device), data.edge_index.to(device), data.batch.to(device)

        # Node-level updates
        x = self.conv1(x, edge_index)
        x = self.relu(x)

        x = self.conv2(x, edge_index)
        x = self.relu(x)

        # Graph-level embedding
        # If batch is missing or None, create a single-graph batch
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        x = global_mean_pool(x, batch)

        return x   # shape: (batch_size, 128)


class GraphTransPooling(nn.Module):

    def __init__(
        self,
        embed_dim=128,
        num_heads=4,
        num_layers=2,
        ff_dim=512,
        output_dim=256,
        dropout=0.1
    ):
        super().__init__()

        # Transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True
        )

        # Stack 2 encoder layers
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Final projection
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, output_dim),
            nn.ReLU()
        )

    def forward(self, x, mask=None):
        """
        x shape:
            (batch_size, num_tokens, embed_dim)

        Example:
            tokens = drug graph nodes + PPI embeddings
        """

        # Transformer self-attention
        x = self.transformer(
            x,
            src_key_padding_mask=mask
        )

        # Mean pooling over tokens
        x = x.mean(dim=1)

        # Final projection
        x = self.fc(x)

        return x
        
        

class Two_D_FEM(nn.Module):
    def __init__(self, ppi_graph_path, drug_targets_path, drug_dbid_mapping_path, drug_smiles_path):
        super().__init__()
        self.ppi = torch.load(ppi_graph_path)
        self.drug_targets = (
            pd.read_csv(drug_targets_path)
            .groupby("DrugID (DrugBank ID)")["Drug_Target (Gene Eentrez IDs)"]
            .apply(list)
            .to_dict()
        )
        # load drug smiles and create mapping: normalized name -> smiles
        df_smiles = pd.read_csv(drug_smiles_path)
        smiles_cols = [c for c in df_smiles.columns if str(c).strip().lower() == 'smiles']
        if smiles_cols:
            name_cols = [c for c in df_smiles.columns if c not in smiles_cols]
            name_col = name_cols[0]
            smiles_col = smiles_cols[0]
            raw_smiles = df_smiles[[name_col, smiles_col]].dropna().set_index(name_col)[smiles_col].to_dict()
        else:
            # Handle headerless files like "drug,smiles" with first row treated as header.
            df_smiles = pd.read_csv(drug_smiles_path, header=None, names=['drug_name', 'smiles'])
            raw_smiles = df_smiles[['drug_name', 'smiles']].dropna().set_index('drug_name')['smiles'].to_dict()
        self.drug_smiles = { normalize_drug_name(k): str(v).strip() for k, v in raw_smiles.items() }

        # load drug name->dbid mapping and normalize keys
        mapping_df = pd.read_csv(drug_dbid_mapping_path)
        mapping_df = mapping_df.dropna(subset=['drugbank_id'])  # Remove rows with NaN drugbank_id
        raw_map = mapping_df.set_index('name_norm')['drugbank_id'].to_dict()
        self.drug_mapping = { normalize_drug_name(k): v for k, v in raw_map.items() }
        self.gcn = MolecularGCN()
        self.graph_trans_pooling = GraphTransPooling()
        self._warned_missing_dbids = set()
        
    def get_ppi_features(self, drug_name): 
        dbid = self.drug_mapping.get(normalize_drug_name(drug_name))
        if dbid is None:
            if drug_name not in self._warned_missing_dbids:
                logging.warning(f"Drug->dbid mapping not found for: {drug_name} - returning empty PPI features")
                self._warned_missing_dbids.add(drug_name)
            device = self.ppi['embeddings'].device
            return torch.empty((0, self.ppi['embeddings'].size(1)), device=device)

        # Check if targets exist for this drug
        if dbid not in self.drug_targets:
            # No targets for this drug, return empty tensor
            device = self.ppi['embeddings'].device
            return torch.empty((0, self.ppi['embeddings'].size(1)), device=device)
    
        targets = self.drug_targets[dbid]
        node_to_idx = self.ppi['node_to_idx']
        valid_targets = [
            t for t in targets
            if str(t) in node_to_idx
        ]
        
        indices = [
            node_to_idx[str(t)]
            for t in valid_targets
        ]
            
        if len(indices) == 0:
            # return empty tensor (no targets) on same device as ppi embeddings
            device = self.ppi['embeddings'].device
            return torch.empty((0, self.ppi['embeddings'].size(1)), device=device)
        return self.ppi["embeddings"][indices]
    
    def forward(self, drug_1, drug_2):
        drug_1_smiles = self.drug_smiles.get(normalize_drug_name(drug_1))
        drug_2_smiles = self.drug_smiles.get(normalize_drug_name(drug_2))
        
        if drug_1_smiles is None or drug_2_smiles is None:
            raise KeyError(f"SMILES not found for: {drug_1} or {drug_2}")

        drug_1_G = smiles_to_graph(drug_1_smiles)
        drug_2_G = smiles_to_graph(drug_2_smiles)

        drug_1_G_embedding = self.gcn(drug_1_G)  # (1, embed_dim)
        drug_2_G_embedding = self.gcn(drug_2_G)

        drug_1_ppi = self.get_ppi_features(drug_1)
        drug_2_ppi = self.get_ppi_features(drug_2)

        # Get device from model
        device = next(self.parameters()).device

        # Ensure embeddings have batch and token dims
        # GCN outputs shape: (batch_size, embed_dim) -> make (batch, 1, embed_dim)
        drug_1_G_embedding = drug_1_G_embedding.unsqueeze(1)
        drug_2_G_embedding = drug_2_G_embedding.unsqueeze(1)

        # PPI features: (num_targets, embed_dim) -> (1, num_targets, embed_dim)
        if drug_1_ppi.numel() == 0:
            drug_1_ppi = torch.zeros((1, 1, self.gcn.conv2.out_channels), dtype=torch.float, device=device)
        else:
            drug_1_ppi = drug_1_ppi.unsqueeze(0).float().to(device)

        if drug_2_ppi.numel() == 0:
            drug_2_ppi = torch.zeros((1, 1, self.gcn.conv2.out_channels), dtype=torch.float, device=device)
        else:
            drug_2_ppi = drug_2_ppi.unsqueeze(0).float().to(device)

        # Concatenate tokens: [drug1_graph, drug1_ppi..., drug2_graph, drug2_ppi...]
        x = torch.cat([drug_1_G_embedding, drug_1_ppi, drug_2_G_embedding, drug_2_ppi], dim=1)

        return self.graph_trans_pooling(x)
        