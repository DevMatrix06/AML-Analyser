import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
import os

# --- Configuration for Elliptic Dataset ---
FEATURES_PATH = 'data/raw/elliptic_txs_features.csv'
EDGES_PATH = 'data/raw/elliptic_txs_edgelist.csv' 
CLASSES_PATH = 'data/raw/elliptic_txs_classes.csv'
PROCESSED_DATA_PATH = 'data/processed/graph_data_bitcoin.pt'

def preprocess_bitcoin_data():
    """
    Loads, merges, and converts the Elliptic dataset files into a PyG Data object.
    Task: Node Classification (predicting if a Bitcoin transaction address is illicit).
    """
    print("1. Loading raw Elliptic data...")

    # Load data: features has no header; edges/classes have headers.
    df_features = pd.read_csv(FEATURES_PATH, header=None)
    df_edges = pd.read_csv(EDGES_PATH)
    df_classes = pd.read_csv(CLASSES_PATH)

    # Correct Column Naming (Based on Elliptic structure)
    df_features.rename(columns={0: 'txId', 1: 'time_step'}, inplace=True)
    
    # CRITICAL FIX: Rename the columns in the edgelist file (txId1 and txId2)
    df_edges.rename(columns={'txId1': 'source', 'txId2': 'target'}, inplace=True) 
    
    # 2. Prepare Node Features (X)
    
    X = df_features.iloc[:, 1:].values 
    X = torch.tensor(X, dtype=torch.float)
    
    # Create a mapping from node ID (txId) to its integer index (0 to N-1)
    node_ids = df_features['txId'].values
    node_map = {txId: i for i, txId in enumerate(node_ids)}
    
    num_nodes = len(node_ids)
    print(f"   -> Total nodes/transactions: {num_nodes:,}")
    print(f"   -> Node feature dimension: {X.size(1)}")
    
    # 3. Prepare Edge Index (Connectivity)
    
    # Map txIds in the edge list to the integer indices
    # Now df_edges has 'source' and 'target' columns with the correct data.
    source_nodes = df_edges['source'].apply(node_map.get)
    target_nodes = df_edges['target'].apply(node_map.get)
    
    # Drop any edges where one or both nodes were missing from the feature set
    valid_edges_df = pd.DataFrame({'source_idx': source_nodes, 'target_idx': target_nodes}).dropna().astype(int)

    source_idx = valid_edges_df['source_idx'].values
    target_idx = valid_edges_df['target_idx'].values
    
    edge_index = torch.tensor([source_idx, target_idx], dtype=torch.long)
    print(f"   -> Total valid edges: {edge_index.size(1):,}")

    # 4. Prepare Node Labels (Y)
    
    # Merge features and classes based on txId
    df_labels = pd.merge(df_features[['txId']], df_classes, on='txId', how='left')
    
    # Map classes: '1' = Illicit (1), '2' = Licit (0), 'unknown' = Unlabeled (-1)
    Y = df_labels['class'].replace({'1': 1, '2': 0, 'unknown': -1}).values
    Y = torch.tensor(Y, dtype=torch.long)
    
    print(f"   -> Illicit/Licit/Unknown Split: {torch.sum(Y==1).item():,}/{torch.sum(Y==0).item():,}/{torch.sum(Y==-1).item():,}")
    
    # 5. Create Masks for Training (Splitting only on Labeled Nodes)
    
    labeled_mask = (Y != -1)
    labeled_indices = torch.where(labeled_mask)[0].numpy()
    Y_labeled = Y[labeled_mask].numpy()
    
    # Stratified split 
    train_idx, temp_idx = train_test_split(labeled_indices, test_size=0.4, stratify=Y_labeled, random_state=42)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, stratify=Y[temp_idx].numpy(), random_state=42)
    
    # Create PyG Masks (size of all nodes)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True
    
    print(f"   -> Labeled nodes (Train/Val/Test): {len(train_idx):,}/{len(val_idx):,}/{len(test_idx):,}")
    
    # 6. Create PyTorch Geometric Data object
    data = Data(
        x=X,
        edge_index=edge_index,
        y=Y, 
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask
    )

    # 7. Save Data
    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
    torch.save(data, PROCESSED_DATA_PATH, _use_new_zipfile_serialization=False)
    print(f"7. ✅ Bitcoin Graph Data saved successfully to: {PROCESSED_DATA_PATH}")

if __name__ == '__main__':
    preprocess_bitcoin_data()