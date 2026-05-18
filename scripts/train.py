import argparse
import os
import random
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import logging
import sys

warnings.filterwarnings(
    "ignore",
    message=r"An issue occurred while importing 'pyg-lib'.*",
)
warnings.filterwarnings(
    "ignore",
    message=r"An issue occurred while importing 'torch-sparse'.*",
)
warnings.filterwarnings(
    "ignore",
    message=r"Deterministic behavior was enabled with either `torch.use_deterministic_algorithms\(True\)`.*",
)
warnings.filterwarnings(
    "ignore",
    message=r"Memory Efficient attention defaults to a non-deterministic algorithm.*",
)

from models import MD_Syn
from models import ONeilDataset, collate_fn
from models.loaders import DrugCombDataset
from torch.utils.data import DataLoader, Subset

# Configure logging to stdout
logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


def seed_everything(seed: int):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Prefer deterministic kernels for reproducibility.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: int):
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def class_balance_from_scores(scores):
    scores = np.asarray(scores, dtype=float)
    class_1 = int(np.sum(scores > 10))
    class_0 = int(np.sum(scores < 0))
    total = class_0 + class_1
    return class_0, class_1, total


def log_class_balance(prefix, scores):
    class_0, class_1, total = class_balance_from_scores(scores)
    if total == 0:
        logging.info(f"{prefix} class balance: no samples")
        return

    pct_0 = 100.0 * class_0 / total
    pct_1 = 100.0 * class_1 / total
    logging.info(
        f"{prefix} class balance: class 0={class_0} ({pct_0:.1f}%), class 1={class_1} ({pct_1:.1f}%), total={total}"
    )


def log_binary_class_balance(prefix, labels):
    """Log binary class balance for datasets with 0/1 labels."""
    labels = np.asarray(labels, dtype=int)
    class_0 = int(np.sum(labels == 0))
    class_1 = int(np.sum(labels == 1))
    total = class_0 + class_1
    
    if total == 0:
        logging.info(f"{prefix} class balance: no samples")
        return
    
    pct_0 = 100.0 * class_0 / total
    pct_1 = 100.0 * class_1 / total
    logging.info(
        f"{prefix} class balance: class 0={class_0} ({pct_0:.1f}%), class 1={class_1} ({pct_1:.1f}%), total={total}"
    )


def compute_metrics(labels, predictions, probabilities=None):
    """
    Compute classification metrics.
    
    Args:
        labels: ground truth class labels (0 or 1)
        predictions: predicted class labels (0 or 1)
        probabilities: probability of class 1 for ROC-AUC (if None, AUROC not computed)
    
    Returns:
        dict with metrics: accuracy, precision, recall, f1, auroc (if probabilities provided)
    """
    metrics = {}
    
    if len(np.unique(labels)) < 2:
        # If only one class present, handle gracefully
        logging.warning(f"Only one class present in labels: {np.unique(labels)}")
        metrics['accuracy'] = accuracy_score(labels, predictions)
        metrics['precision'] = 0.0
        metrics['recall'] = 0.0
        metrics['f1'] = 0.0
        metrics['auroc'] = 0.0
        return metrics
    
    metrics['accuracy'] = accuracy_score(labels, predictions)
    metrics['precision'] = precision_score(labels, predictions, zero_division=0)
    metrics['recall'] = recall_score(labels, predictions, zero_division=0)
    metrics['f1'] = f1_score(labels, predictions, zero_division=0)
    
    if probabilities is not None:
        try:
            metrics['auroc'] = roc_auc_score(labels, probabilities)
        except Exception as e:
            logging.warning(f"Could not compute AUROC: {e}")
            metrics['auroc'] = 0.0
    
    return metrics


def log_metrics(prefix, metrics):
    """Log computed metrics in a readable format."""
    msg = f"{prefix} - "
    msg += f"Acc: {metrics['accuracy']:.4f}, "
    msg += f"Prec: {metrics['precision']:.4f}, "
    msg += f"Rec: {metrics['recall']:.4f}, "
    msg += f"F1: {metrics['f1']:.4f}"
    if 'auroc' in metrics:
        msg += f", AUROC: {metrics['auroc']:.4f}"
    logging.info(msg)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    batches = 0
    total_samples = 0

    for batch_idx, batch in enumerate(loader):
        drug_1_list = batch['drug_1']
        drug_2_list = batch['drug_2']
        cell_line_list = batch['cell_line']
        labels = batch['label'].to(device)
        
        batch_size = len(labels)
        total_samples += batch_size

        outputs = []
        for d1, d2, cl in zip(drug_1_list, drug_2_list, cell_line_list):
            out = model(d1, d2, cl)
            # ensure shape (1, output_dim)
            if out.dim() == 1:
                out = out.unsqueeze(0)
            outputs.append(out)

        outputs = torch.cat(outputs, dim=0).to(device)  # (batch, num_classes)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        batches += 1
        
        if (batch_idx + 1) % 10 == 0:
            logging.info(f"  Batch {batch_idx + 1}: {total_samples} samples processed, Loss: {loss.item():.4f}")

    avg_loss = total_loss / batches if batches > 0 else 0.0
    logging.info(f"Epoch Summary: {total_samples} total samples, {batches} batches, Avg Loss: {avg_loss:.4f}")
    return avg_loss


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    batches = 0
    total_samples = 0
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            drug_1_list = batch['drug_1']
            drug_2_list = batch['drug_2']
            cell_line_list = batch['cell_line']
            labels = batch['label'].to(device)
            
            batch_size = len(labels)
            total_samples += batch_size

            outputs = []
            for d1, d2, cl in zip(drug_1_list, drug_2_list, cell_line_list):
                out = model(d1, d2, cl)
                if out.dim() == 1:
                    out = out.unsqueeze(0)
                outputs.append(out)

            outputs = torch.cat(outputs, dim=0).to(device)  # (batch, 2)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            batches += 1
            
            # Get predictions and probabilities
            probs = torch.softmax(outputs, dim=1)  # (batch, 2)
            preds = torch.argmax(outputs, dim=1)  # (batch,)
            
            all_predictions.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probs[:, 1].cpu().numpy())  # prob of class 1

    avg_loss = total_loss / batches if batches > 0 else 0.0
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probabilities = np.array(all_probabilities)
    
    return avg_loss, all_labels, all_predictions, all_probabilities, total_samples


def run_cross_validation(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    dataset = ONeilDataset(args.labels_path)
    logging.info(f"Loaded dataset with {len(dataset)} total samples")
    log_class_balance("Overall dataset", dataset.labels_df['synergy'])
    
    n_splits = 5
    indices = np.arange(len(dataset))

    try:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=args.seed)
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
        logging.info(f"=== Fold {fold}/{n_splits} ===")
        logging.info(f"  Train samples: {len(train_idx)}, Val samples: {len(val_idx)}")
        log_class_balance(f"Fold {fold} train", dataset.labels_df.iloc[train_idx]['synergy'])
        log_class_balance(f"Fold {fold} val", dataset.labels_df.iloc[val_idx]['synergy'])

        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        train_loader = DataLoader(
            train_subset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            worker_init_fn=seed_worker,
            generator=make_generator(args.seed + fold)
        )
        val_loader = DataLoader(
            val_subset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            worker_init_fn=seed_worker,
            generator=make_generator(args.seed + fold)
        )

        model = MD_Syn(
            args.cell_line_embeddings_path,
            args.cell_line_mapping_path,
            args.landmarks_path,
            args.molformer_embeddings_path,
            args.ppi_graph_path,
            args.drug_targets_path,
            args.drug_dbid_mapping_path,
            args.drug_smiles_path,
            use_drugcomb=False
        ).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

        for epoch in range(args.num_epochs):
            logging.info(f"Fold {fold} - Epoch {epoch+1}/{args.num_epochs}")
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
            
            val_loss, val_labels, val_preds, val_probs, val_samples = evaluate(model, val_loader, criterion, device)
            val_metrics = compute_metrics(val_labels, val_preds, val_probs)
            
            logging.info(f"Fold {fold} Epoch {epoch+1}/{args.num_epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
            log_metrics(f"  Fold {fold} Val", val_metrics)

        logging.info(f"Fold {fold} training complete. Saving model to model_fold{fold}.pt")
        torch.save(model.state_dict(), f"model_fold{fold}.pt")


def run_cross_validation_drugcomb(args):
    """Run n-fold cross-validation using DrugComb dataset with pre-assigned folds."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    # Load full dataset to get unique fold information
    full_dataset = DrugCombDataset(args.labels_path)
    logging.info(f"Loaded full DrugComb dataset with {len(full_dataset)} total samples")
    log_binary_class_balance("Overall dataset", full_dataset.labels_df['label'])
    
    unique_folds = sorted(full_dataset.labels_df['fold'].unique())
    n_folds = len(unique_folds)
    logging.info(f"Found {n_folds} folds in dataset: {unique_folds}")

    for fold_idx, val_fold in enumerate(unique_folds, 1):
        logging.info(f"=== Fold {fold_idx}/{n_folds} (validation fold={val_fold}) ===")
        
        # Create train and validation subsets
        train_folds = [f for f in unique_folds if f != val_fold]
        
        train_dataset = DrugCombDataset(args.labels_path, folds=train_folds)
        val_dataset = DrugCombDataset(args.labels_path, folds=[val_fold])
        
        logging.info(f"  Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
        log_binary_class_balance(f"Fold {fold_idx} train", train_dataset.labels_df['label'])
        log_binary_class_balance(f"Fold {fold_idx} val", val_dataset.labels_df['label'])

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            worker_init_fn=seed_worker,
            generator=make_generator(args.seed + fold_idx)
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            worker_init_fn=seed_worker,
            generator=make_generator(args.seed + fold_idx)
        )

        model = MD_Syn(
            args.cell_line_embeddings_path,
            args.cell_line_mapping_path,
            args.landmarks_path,
            args.molformer_embeddings_path,
            args.ppi_graph_path,
            args.drug_targets_path,
            args.drug_dbid_mapping_path,
            args.drug_smiles_path,
            use_drugcomb=True
        ).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

        for epoch in range(args.num_epochs):
            logging.info(f"Fold {fold_idx} - Epoch {epoch+1}/{args.num_epochs}")
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
            
            val_loss, val_labels, val_preds, val_probs, val_samples = evaluate(model, val_loader, criterion, device)
            val_metrics = compute_metrics(val_labels, val_preds, val_probs)
            
            logging.info(f"Fold {fold_idx} Epoch {epoch+1}/{args.num_epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
            log_metrics(f"  Fold {fold_idx} Val", val_metrics)

        logging.info(f"Fold {fold_idx} training complete. Saving model to model_drugcomb_fold{val_fold}.pt")
        torch.save(model.state_dict(), f"model_drugcomb_fold{val_fold}.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='oneil', choices=['oneil', 'drugcomb'],
                        help='Dataset to use: oneil or drugcomb')
    parser.add_argument('--labels_path', type=str, required=False,
                        default="/home/xuzijie/MD-syn-GEMS/data/labels.csv")
    parser.add_argument('--cell_line_embeddings_path', type=str, default="/home/xuzijie/MD-syn-GEMS/data/CCLE_expression.csv")
    parser.add_argument('--cell_line_mapping_path', type=str, default="/home/xuzijie/MD-syn-GEMS/data/cell_line_ccle_map.csv")
    parser.add_argument('--landmarks_path', type=str, default="/home/xuzijie/MD-syn-GEMS/data/landmark_genes.txt")
    parser.add_argument('--molformer_embeddings_path', type=str, default="/home/xuzijie/MD-syn-GEMS/data/molformer_embeddings.pt")
    parser.add_argument('--ppi_graph_path', type=str, default="/home/xuzijie/MD-syn-GEMS/data/ppi_node2vec_matrix.pt")
    parser.add_argument('--drug_targets_path', type=str, default="/home/xuzijie/MD-syn-GEMS/data/drug_target.csv")
    parser.add_argument('--drug_dbid_mapping_path', type=str, default="/home/xuzijie/MD-syn-GEMS/data/drugcomb_name_to_dbid_mapping_improved.csv")
    parser.add_argument('--drug_smiles_path', type=str, default="/home/xuzijie/MD-syn-GEMS/data/smiles.csv")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=5e-4)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cv', action='store_true', help='Run cross-validation (5-fold for oneil, folds in dataset for drugcomb)')

    args = parser.parse_args()
    # pass some args through
    args.cell_line_embeddings_path = args.cell_line_embeddings_path
    args.cell_line_mapping_path = args.cell_line_mapping_path
    args.landmarks_path = args.landmarks_path
    args.molformer_embeddings_path = args.molformer_embeddings_path
    args.ppi_graph_path = args.ppi_graph_path
    args.drug_targets_path = args.drug_targets_path
    args.drug_dbid_mapping_path = args.drug_dbid_mapping_path
    args.drug_smiles_path = args.drug_smiles_path
    args.batch_size = args.batch_size
    args.learning_rate = args.learning_rate
    args.num_epochs = args.num_epochs
    args.seed = args.seed

    seed_everything(args.seed)
    
    logging.info(f'DATASET: {args.dataset}')
    logging.info(f'CELL LINE EMBEDDING FILE: {args.cell_line_embeddings_path}')
    logging.info(f'CELL LINE MAPPING FILE: {args.cell_line_mapping_path}')
    logging.info(f'LANDMARKS FILE: {args.landmarks_path}')
    logging.info(f'MOLFORMER EMBEDDINGS FILE: {args.molformer_embeddings_path}')
    logging.info(f'PPI GRAPH FILE: {args.ppi_graph_path}')
    logging.info(f'DRUG TARGETS FILE: {args.drug_targets_path}')
    logging.info(f'DBID MAPPING FILE: {args.drug_dbid_mapping_path}')
    logging.info(f'SMILES FILE: {args.drug_smiles_path}')
    logging.info(f'BATCH SIZE: {args.batch_size}')
    logging.info(f'LEARNING RATE: {args.learning_rate}')
    logging.info(f'NUM EPOCHS: {args.num_epochs}')
    logging.info(f'SEED: {args.seed}')
    

    if args.cv:
        logging.info('=== USING CROSS VALIDATION ===')
        if args.dataset == 'drugcomb':
            run_cross_validation_drugcomb(args)
        else:
            run_cross_validation(args)
    else:
        logging.info('=== SINGLE RUN ===')
        logging.info(f"Using device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
        # Single-run training using entire dataset
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if args.dataset == 'drugcomb':
            dataset = DrugCombDataset(args.labels_path)
            log_fn = log_binary_class_balance
            use_drugcomb = True
        else:
            dataset = ONeilDataset(args.labels_path)
            log_fn = log_class_balance
            use_drugcomb = False
            
        logging.info(f"Loaded dataset with {len(dataset)} samples")
        log_fn("Overall dataset", dataset.labels_df['label'] if args.dataset == 'drugcomb' else dataset.labels_df['synergy'])
        
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            worker_init_fn=seed_worker,
            generator=make_generator(args.seed)
        )

        model = MD_Syn(
            args.cell_line_embeddings_path,
            args.cell_line_mapping_path,
            args.landmarks_path,
            args.molformer_embeddings_path,
            args.ppi_graph_path,
            args.drug_targets_path,
            args.drug_dbid_mapping_path,
            args.drug_smiles_path,
            use_drugcomb=use_drugcomb
        ).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

        for epoch in range(args.num_epochs):
            logging.info(f"Epoch {epoch+1}/{args.num_epochs}")
            train_loss = train_one_epoch(model, loader, criterion, optimizer, device)
            logging.info(f"Epoch {epoch+1}/{args.num_epochs} - Train Loss: {train_loss:.4f}")

        # Compute final training metrics
        train_loss, train_labels, train_preds, train_probs, _ = evaluate(model, loader, criterion, device)
        train_metrics = compute_metrics(train_labels, train_preds, train_probs)
        log_metrics("Final Training", train_metrics)

        logging.info("Training complete. Saving model to model.pt")
        torch.save(model.state_dict(), "model.pt")