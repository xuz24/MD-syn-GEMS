import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class ONeilDataset(Dataset):
    """
    ONeil drug synergy dataset loader.
    Loads pairs of drugs with cell lines and their synergy labels.
    Deduplicates by (drug_a_name, drug_b_name, cell_line) triplet, averaging synergy scores,
    then applies binary threshold to the averaged scores.
    """
    def __init__(self, labels_path):
        """
        Args:
            labels_path: Path to labels.csv with columns: drug_a_name, drug_b_name, cell_line, synergy
            
        Deduplication strategy:
            1. Load all rows and drop NAs
            2. Group by triplet (drug_a_name, drug_b_name, cell_line)
            3. Average synergy for each triplet
            4. Apply strict binary threshold to averaged scores:
               synergy > 10 -> 1 (synergistic)
               synergy < 0 -> 0 (antagonistic)
               0 <= synergy <= 10 rows are dropped.
        """
        self.labels_df = pd.read_csv(labels_path)
        self.labels_df = self.labels_df.dropna(subset=['drug_a_name', 'drug_b_name', 'cell_line', 'synergy'])
        
        # Deduplicate by triplet: group and average synergy scores
        self.labels_df = self.labels_df.groupby(
            ['drug_a_name', 'drug_b_name', 'cell_line'],
            as_index=False
        )['synergy'].mean()
        
        # Keep only strict binary classes after averaging.
        self.labels_df = self.labels_df[(self.labels_df['synergy'] > 10) | (self.labels_df['synergy'] < 0)].copy()
        
    def __len__(self):
        return len(self.labels_df)
    
    def __getitem__(self, idx):
        row = self.labels_df.iloc[idx]
        drug_1 = str(row['drug_a_name']).strip()
        drug_2 = str(row['drug_b_name']).strip()
        cell_line = str(row['cell_line']).strip()
        label = 1 if float(row['synergy']) > 10 else 0
        
        return {
            'drug_1': drug_1,
            'drug_2': drug_2,
            'cell_line': cell_line,
            'label': torch.tensor(label, dtype=torch.long)
        }


def collate_fn(batch):
    """
    Custom collate function for DataLoader.
    Groups batch items for processing with MD_Syn model.
    
    Args:
        batch: List of dicts from ONeilDataset
    
    Returns:
        dict with lists of drug pairs, cell lines, and stacked label tensor
    """
    drug_1_list = [item['drug_1'] for item in batch]
    drug_2_list = [item['drug_2'] for item in batch]
    cell_line_list = [item['cell_line'] for item in batch]
    labels = torch.stack([item['label'] for item in batch])
    folds = [item['fold'] for item in batch] if 'fold' in batch[0] else None

    collated = {
        'drug_1': drug_1_list,
        'drug_2': drug_2_list,
        'cell_line': cell_line_list,
        'label': labels
    }

    if folds is not None:
        collated['fold'] = torch.tensor(folds, dtype=torch.long)

    return collated


def get_dataloader(labels_path, batch_size=32, shuffle=True, num_workers=0):
    """
    Create a DataLoader for the ONeil dataset.
    
    Args:
        labels_path: Path to labels.csv
        batch_size: Batch size (default 32)
        shuffle: Whether to shuffle (default True)
        num_workers: Number of workers for data loading (default 0)
    
    Returns:
        torch.utils.data.DataLoader
    """
    dataset = ONeilDataset(labels_path)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn
    )

class DrugCombDataset(Dataset):
    """
    DrugComb fold-assigned dataset loader.

    The canonical CSV uses binary labels (0/1) and a fold column with values 0, 1, 2.
    Optionally filter to a subset of folds by passing `folds`.
    """

    def __init__(self, labels_path, folds=None):
        self.labels_df = pd.read_csv(labels_path)
        self.labels_df = self.labels_df.dropna(subset=['drug1_dbid', 'drug2_dbid', 'cell_name', 'label', 'fold']).copy()

        self.labels_df['label'] = self.labels_df['label'].astype(int)
        self.labels_df['fold'] = self.labels_df['fold'].astype(int)

        if folds is not None:
            if isinstance(folds, int):
                folds = [folds]
            self.labels_df = self.labels_df[self.labels_df['fold'].isin(folds)].copy()

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        row = self.labels_df.iloc[idx]
        drug_1 = str(row['drug1_dbid']).strip()
        drug_2 = str(row['drug2_dbid']).strip()
        cell_line = str(row['cell_name']).strip()
        label = int(row['label'])
        fold = int(row['fold'])

        return {
            'drug_1': drug_1,
            'drug_2': drug_2,
            'cell_line': cell_line,
            'label': torch.tensor(label, dtype=torch.long),
            'fold': fold,
        }