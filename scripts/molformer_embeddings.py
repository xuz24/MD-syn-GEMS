from transformers import AutoTokenizer, AutoModel
import torch
import pandas as pd


tokenizer = AutoTokenizer.from_pretrained(
    "ibm/MoLFormer-XL-both-10pct",
    trust_remote_code=True
)

model = AutoModel.from_pretrained(
    "ibm/MoLFormer-XL-both-10pct",
    trust_remote_code=True
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model.to(device)
model.eval()


def get_molformer_embedding(smiles: str) -> torch.Tensor:

    tokens = tokenizer(
        smiles,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(device)

    with torch.no_grad():
        output = model(**tokens)

        return (
            output.last_hidden_state[:, 0, :]
            .squeeze(0)
            .detach()
            .cpu()
            .float()
        )


if __name__ == "__main__":

    embeddings_dict = {}

    data_file = "/home/xuzijie/MD-syn-GEMS/data/smiles.csv"
    
    print(f'=== SMILES FILE: {data_file} ===')

    def _normalize(name: str) -> str:
        return str(name).strip().lower()

    df = pd.read_csv(data_file, header=None, names=["drug", "smiles"])

    print('=== START PROCESSING ===')
    
    for _, row in df.iterrows():

        drug = row["drug"]
        smiles = row["smiles"]
        
        print(f"Processing {drug}")

        if pd.isna(smiles):
            print(f'No SMILES found for {drug}')
            continue

        embeddings_dict[_normalize(drug)] = (
            get_molformer_embedding(smiles)
        )

    torch.save(
        embeddings_dict,
        "molformer_embeddings.pt"
    )