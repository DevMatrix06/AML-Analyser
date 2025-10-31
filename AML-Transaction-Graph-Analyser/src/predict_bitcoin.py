import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import os
from torch_geometric.data import Data 
from model_bitcoin import BitcoinGraphSAGE # Import the new model

# --- Configuration (UPDATED PATHS) ---
PROCESSED_DATA_PATH = 'data/processed/graph_data_bitcoin.pt'
MODEL_CHECKPOINT_PATH = 'outputs/model_checkpoint_bitcoin.pth'
OUTPUT_SCORES_PATH = 'outputs/bitcoin_predictions.csv'
FEATURES_PATH = 'data/raw/elliptic_txs_features.csv' # Used to get txId
CLASSES_PATH = 'data/raw/elliptic_txs_classes.csv' # Used to get ground truth

# --- Hyperparameters (Must match train_bitcoin.py) ---
try:
    data = torch.load(PROCESSED_DATA_PATH, weights_only=False)
except:
    exit()

IN_CHANNELS_X = data.x.size(1)             
HIDDEN_CHANNELS = 128                      
OUT_CHANNELS_NODE = 64

def generate_bitcoin_predictions():
    """Loads the model and generates illicit probability scores for all addresses."""
    print("1. Loading Model and Data...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize the model structure
    model = BitcoinGraphSAGE(IN_CHANNELS_X, HIDDEN_CHANNELS, OUT_CHANNELS_NODE).to(device)
    
    # Load the trained weights
    model.load_state_dict(torch.load(MODEL_CHECKPOINT_PATH, map_location=device))
    model.eval()
    data.to(device)
    
    # --- 2. Generating Scores ---
    print("2. Running inference on all Bitcoin addresses...")
    with torch.no_grad():
        # raw_scores shape: N_NODES x 2 (Licit/Illicit logits)
        raw_scores, _ = model(data)
        
        # Convert logits to probabilities (Softmax)
        probabilities = F.softmax(raw_scores, dim=1)
        
        # The score of interest is the probability of class 1 (Illicit)
        suspicion_scores = probabilities[:, 1].cpu().numpy() 

    # --- 3. Merging with Original Data ---
    print("3. Merging scores with original Bitcoin transaction IDs...")
    
    # Load original feature and class files to link indices back to txIds
    df_features = pd.read_csv(FEATURES_PATH, header=None)
    df_features.rename(columns={0: 'txId', 1: 'TimeStep'}, inplace=True)
    df_classes = pd.read_csv(CLASSES_PATH)

    # Create the prediction DataFrame
    df_pred = pd.DataFrame({
        'txId': df_features['txId'].values,
        'SUSPICION_SCORE': suspicion_scores
    })
    
    # Merge with original classes to get TimeStep and Ground Truth
    df_output = pd.merge(df_pred, df_classes, on='txId', how='left')
    df_output = pd.merge(df_output, df_features[['txId', 'TimeStep']], on='txId', how='left')
    
    # Clean up classes for display
    df_output['class'] = df_output['class'].replace({'1': 'Illicit', '2': 'Licit', 'unknown': 'Unlabeled'})
    
    # --- 4. Saving Results ---
    os.makedirs('outputs', exist_ok=True)
    df_output.to_csv(OUTPUT_SCORES_PATH, index=False)
    
    print(f"4. ✅ Bitcoin Prediction scores saved to: {OUTPUT_SCORES_PATH}")
    
    # Display top 5 most suspicious nodes
    top_5 = df_output.sort_values(by='SUSPICION_SCORE', ascending=False).head(5)
    print("\nTop 5 Most Suspicious Bitcoin Addresses:")
    print(top_5[['txId', 'TimeStep', 'class', 'SUSPICION_SCORE']].to_markdown(index=False))
    
    print("\nBitcoin Prediction pipeline complete.")

if __name__ == "__main__":
    if os.path.exists(MODEL_CHECKPOINT_PATH):
        generate_bitcoin_predictions()
    else:
        print(f"Error: Model checkpoint not found at {MODEL_CHECKPOINT_PATH}. Run train_bitcoin.py first.")
