from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
import pandas as pd


tokenizer = AutoTokenizer.from_pretrained("ibm/MoLFormer-XL-both-10pct", trust_remote_code=True)
model = AutoModel.from_pretrained("ibm/MoLFormer-XL-both-10pct", trust_remote_code=True)
model.eval()

def get_molformer_embedding(smiles: str) -> torch.Tensor:
    tokens = tokenizer(smiles, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        output = model(**tokens)
        return output.last_hidden_state[:, 0, :].squeeze(0).detach().cpu().float() # CLS token, shape (768, )

if __name__ == "__main__":
    embeddings_dict = {}
    
    data_file = "/Users/zijiexu/GEMS-LAB/jumpstart_bundle 2/data/synergy_labels/oneil_dataset/smiles.csv" # smiles file
    def _normalize(name: str) -> str:
        return str(name).strip().lower()

    with open(data_file) as f:
        df = pd.read_csv(data_file)

        for _, row in df.iterrows():
            drug = row["drug"]
            smiles = row["smiles"]

            embeddings_dict[_normalize(drug)] = get_molformer_embedding(smiles)
    
    torch.save(embeddings_dict, "molformer_embeddings.pt")