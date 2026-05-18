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

    dbid_file = "/home/xuzijie/MD-syn-GEMS/data/canonical_folds.csv"
    smiles_file = "/home/xuzijie/MD-syn-GEMS/data/molecules.csv"

    
    print(f'=== SMILES FILES: {smiles_file} ===')
    print(f"=== DBIDS FROM: {dbid_file} ===\n")


    def _normalize(name: str) -> str:
        return str(name).strip().lower()

    fold_splits = pd.read_csv(dbid_file)
    smiles = pd.read_csv(smiles_file).set_index('drugbank_id')['smiles'].dropna()
    
    
    dbids = dbids = (
        set(fold_splits['drug1_dbid'].dropna().unique()) |
        set(fold_splits['drug2_dbid'].dropna().unique())
    )
    print('=== START PROCESSING ===')
    
    count = 0
    missing = 0
    
    for id in dbids:
        smilesString = smiles.get(id)
        if smilesString is None or pd.isna(smilesString):
            missing += 1
            print(f'{id} missing')
        else:
            embeddings_dict[id] = get_molformer_embedding(smilesString)
            
        count += 1
        
        if count % 100 == 0:
            print(f'=== PROCESSED {count} entries ===')

    print('=== SAVING TO: molformer_embeddings.pt ===')
    torch.save(
        embeddings_dict,
        "molformer_embeddings.pt"
    )
    
    print("=== DONE PROCESSSING ===")
    print('=== SUMMARY ===')
    print(f'total: {count}')
    print(f'missing: {missing}')
    print(f'final total: {count - missing}')
    