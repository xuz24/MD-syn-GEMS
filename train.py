import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold

from models import MD_Syn
from data import ONeilDataset, collate_fn
from torch.utils.data import DataLoader, Subset


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    batches = 0

    for batch in loader:
        drug_1_list = batch['drug_1']
        drug_2_list = batch['drug_2']
        cell_line_list = batch['cell_line']
        labels = batch['label'].to(device)

        outputs = []
        for d1, d2, cl in zip(drug_1_list, drug_2_list, cell_line_list):
            out = model(d1, d2, cl)
            # ensure shape (1, output_dim)
            if out.dim() == 1:
                out = out.unsqueeze(0)
            outputs.append(out)

        outputs = torch.cat(outputs, dim=0).to(device)  # (batch, output_dim)
        # reduce to scalar prediction if needed
        preds = outputs.squeeze() if outputs.dim() == 1 else outputs[:, 0]

        loss = criterion(preds, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        batches += 1

    return total_loss / batches if batches > 0 else 0.0


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    batches = 0
    with torch.no_grad():
        for batch in loader:
            drug_1_list = batch['drug_1']
            drug_2_list = batch['drug_2']
            cell_line_list = batch['cell_line']
            labels = batch['label'].to(device)

            outputs = []
            for d1, d2, cl in zip(drug_1_list, drug_2_list, cell_line_list):
                out = model(d1, d2, cl)
                if out.dim() == 1:
                    out = out.unsqueeze(0)
                outputs.append(out)

            outputs = torch.cat(outputs, dim=0).to(device)
            preds = outputs.squeeze() if outputs.dim() == 1 else outputs[:, 0]

            loss = criterion(preds, labels)
            total_loss += loss.item()
            batches += 1

    return total_loss / batches if batches > 0 else 0.0


def run_cross_validation(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = ONeilDataset(args.labels_path)
    n_splits = 5
    indices = np.arange(len(dataset))

    try:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        splits = list(kf.split(indices))
    except Exception:
        # fallback: simple numpy split
        splits = []
        parts = np.array_split(indices, n_splits)
        for i in range(n_splits):
            val_idx = parts[i]
            train_idx = np.concatenate([parts[j] for j in range(n_splits) if j != i])
            splits.append((train_idx, val_idx))

    for fold, (train_idx, val_idx) in enumerate(splits, 1):
        print(f"=== Fold {fold}/{n_splits} ===")

        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

        model = MD_Syn(
            args.cell_line_embeddings_path,
            args.landmarks_path,
            args.molformer_embeddings_path,
            args.ppi_graph_path,
            args.drug_targets_path,
            args.drug_dbid_mapping_path,
            args.drug_smiles_path,
        ).to(device)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

        for epoch in range(args.num_epochs):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss = evaluate(model, val_loader, criterion, device)
            print(f"Fold {fold} Epoch {epoch+1}/{args.num_epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

        torch.save(model.state_dict(), f"model_fold{fold}.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--labels_path', type=str, required=False,
                        default="/Users/zijiexu/GEMS-LAB/jumpstart_bundle 2/data/synergy_labels/oneil_dataset/labels.csv")
    parser.add_argument('--cell_line_embeddings_path', type=str, default="data/CCLE_expression.csv")
    parser.add_argument('--landmarks_path', type=str, default="data/landmark_genes.txt")
    parser.add_argument('--molformer_embeddings_path', type=str, default="molformer_embeddings.pt")
    parser.add_argument('--ppi_graph_path', type=str, default="ppi_node2vec_matrix.pt")
    parser.add_argument('--drug_targets_path', type=str, default="/Users/zijiexu/GEMS-LAB/jumpstart_bundle 2/data/protein_features/drug_target.csv")
    parser.add_argument('--drug_dbid_mapping_path', type=str, default="/Users/zijiexu/GEMS-LAB/jumpstart_bundle 2/data/synergy_labels/drugcomb_name_to_dbid_mapping.csv")
    parser.add_argument('--drug_smiles_path', type=str, default="/Users/zijiexu/GEMS-LAB/jumpstart_bundle 2/data/synergy_labels/oneil_dataset/smiles.csv")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--cv', action='store_true', help='Run 5-fold cross-validation')

    args = parser.parse_args()
    # pass some args through
    args.cell_line_embeddings_path = args.cell_line_embeddings_path
    args.landmarks_path = args.landmarks_path
    args.molformer_embeddings_path = args.molformer_embeddings_path
    args.ppi_graph_path = args.ppi_graph_path
    args.drug_targets_path = args.drug_targets_path
    args.drug_dbid_mapping_path = args.drug_dbid_mapping_path
    args.drug_smiles_path = args.drug_smiles_path
    args.batch_size = args.batch_size
    args.learning_rate = args.learning_rate
    args.num_epochs = args.num_epochs

    if args.cv:
        run_cross_validation(args)
    else:
        # Single-run training using entire dataset
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        dataset = ONeilDataset(args.labels_path)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

        model = MD_Syn(
            args.cell_line_embeddings_path,
            args.landmarks_path,
            args.molformer_embeddings_path,
            args.ppi_graph_path,
            args.drug_targets_path,
            args.drug_dbid_mapping_path,
            args.drug_smiles_path,
        ).to(device)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

        for epoch in range(args.num_epochs):
            train_loss = train_one_epoch(model, loader, criterion, optimizer, device)
            print(f"Epoch {epoch+1}/{args.num_epochs} - Train Loss: {train_loss:.4f}")

        torch.save(model.state_dict(), "model.pt")
        print("Training complete; model saved to model.pt")