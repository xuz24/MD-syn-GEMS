import torch
import torch.nn as nn
from one_d_fem import One_D_FEM 
from two_d_fem import Two_D_FEM
from ddi_predictor import DDI_Predictor

class MD_Syn(nn.Module):
    def __init__(self, 
                 cell_line_embeddings_path, landmarks_path, molformer_embeddings_path, 
                 ppi_graph_path, drug_targets_path, drug_dbid_mapping_path, drug_smiles_path
                 ):
        super().__init__()
        self.one_d_fem = One_D_FEM(cell_line_embeddings_path, landmarks_path, molformer_embeddings_path)
        self.two_d_fem = Two_D_FEM(ppi_graph_path, drug_targets_path, drug_dbid_mapping_path, drug_smiles_path)
        self.ddi_predictor = DDI_Predictor()
        
    def forward(self, drug_1, drug_2, cell_line):
        one_d_embeddings = self.one_d_fem(drug_1, drug_2, cell_line)
        two_d_embeddings = self.two_d_fem(drug_1, drug_2)
        
        combined = torch.cat([one_d_embeddings, two_d_embeddings], dim=1)
        
        return self.ddi_predictor(combined)