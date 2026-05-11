from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
import pandas as pd


class CellLineEncoder(nn.Module):
    def __init__(self, input_dim=978, embed_dim=256):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, embed_dim)
        )

    def forward(self, x):
        return self.net(x)

class One_D_FEM(nn.Module):
    def __init__(self, cell_line_embeddings_path, landmarks_path, molformer_embeddings_path):
        super().__init__()
        self.cell_line_embeddings = pd.read_csv(cell_line_embeddings_path)
        self.landmarks = pd.read_csv(landmarks_path, sep="\t")
        self.init_cell_embeddings()
        # load molformer embeddings and normalize keys
        def _normalize(name: str) -> str:
            return str(name).strip().lower()

        raw = torch.load(molformer_embeddings_path)
        # if it's a dict-like mapping, normalize keys
        try:
            self.molformer_embeddings = { _normalize(k): v for k, v in raw.items() }
        except Exception:
            # fallback: keep as-is
            self.molformer_embeddings = raw
        self.cell_line_encoder = CellLineEncoder()
        
    def init_cell_embeddings(self):
        landmark_ids = list(self.landmarks["Entrez ID"])
        landmark_ids = [str(id) for id in landmark_ids]
        
        id_to_col = {}

        for col in self.cell_line_embeddings.columns:
            id = col.split()[-1][1:-1]
            id_to_col[id] = col
        
        col_names = []
        for id in landmark_ids:
            col_names.append(id_to_col[id])
        col_names.append('Unnamed: 0')
        

        self.cell_line_embeddings = self.cell_line_embeddings[col_names]
        self.cell_line_embeddings.columns = [col.split()[-1][1:-1] for col in self.cell_line_embeddings.columns]
        self.cell_line_embeddings = self.cell_line_embeddings.rename(columns={"": "cell_line"})
        self.cell_line_embeddings = self.cell_line_embeddings.set_index('cell_line')

    
    def get_molformer_embedding(self, drug: str) -> torch.Tensor:
        def _normalize(name: str) -> str:
            return str(name).strip().lower()

        key = _normalize(drug)
        emb = self.molformer_embeddings[key]
        if isinstance(emb, torch.Tensor):
            if emb.dim() == 1:
                emb = emb.unsqueeze(0)
            return emb.float()
        # If stored as numpy array or list
        emb = torch.tensor(emb, dtype=torch.float32)
        if emb.dim() == 1:
            emb = emb.unsqueeze(0)
        return emb
    
    def forward(self, drug_1, drug_2, cell_line):
        drug_1_embedding = self.get_molformer_embedding(drug_1)
        drug_2_embedding = self.get_molformer_embedding(drug_2)
        cell_line_embedding = torch.tensor(
            self.cell_line_embeddings.loc[cell_line].values,
            dtype=torch.float32
        ).unsqueeze(0)
        cell_line_embedding = self.cell_line_encoder(cell_line_embedding)
        
        # ensure all tensors have batch dim as first dim
        if drug_1_embedding.dim() == 1:
            drug_1_embedding = drug_1_embedding.unsqueeze(0)
        if drug_2_embedding.dim() == 1:
            drug_2_embedding = drug_2_embedding.unsqueeze(0)

        # move embeddings to same dtype/device as cell_line_embedding
        device = cell_line_embedding.device
        drug_1_embedding = drug_1_embedding.to(device)
        drug_2_embedding = drug_2_embedding.to(device)

        return torch.cat([drug_1_embedding, drug_2_embedding, cell_line_embedding], dim=1)
        
