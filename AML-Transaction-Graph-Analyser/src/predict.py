import torch
import pandas as pd
import numpy as np
import os
from torch_geometric.data import Data # Import to ensure correct loading of graph data

# Import the model definition from your model.py file
from model import AMLGraphSAGE 

# --- Configuration ---
PROCESSED_DATA_PATH = 'data/processed/graph_data.pt'
MODEL_CHECKPOINT_PATH = 'outputs/model_checkpoint.pth'
OUTPUT_SCORES_PATH = 'outputs/suspicion_scores.csv'
RAW_DATA_PATH = 'data/raw/amlsim_transactions.csv' # Need raw data for account IDs

# --- Hyperparameters (Must match train.py) ---
# Load data just to extract dimensions (using the necessary weights_only=False fix)
try:
    data = torch.load(PROCESSED_DATA_PATH, weights_only=False)
except Exception:
    # If the safe load fails, try with a simplified version for dimensions only
    # (This is a safety net, but the first line should work if training worked)
    data = torch.load(PROCESSED_DATA_PATH, weights_only=False) 
    
IN_CHANNELS_X = data.x.size(1)              
IN_CHANNELS_EDGE = data.edge_attr.size(1)   
HIDDEN_CHANNELS = 64
OUT_CHANNELS_NODE = 32
# --- End Hyperparameters ---


def generate_predictions():
    """Loads the model and generates suspicion scores for all transactions."""
    print("1. Loading Model and Data...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize the model structure
    model = AMLGraphSAGE(IN_CHANNELS_X, IN_CHANNELS_EDGE, HIDDEN_CHANNELS, OUT_CHANNELS_NODE).to(device)
    
    # Load the trained weights
    model.load_state_dict(torch.load(MODEL_CHECKPOINT_PATH, map_location=device))
    model.eval()
    data.to(device)
    
    # --- 2. Generating Scores ---
    print("2. Running inference on all transactions...")
    with torch.no_grad():
        # Run forward pass on the entire graph
        # We only care about the second return value: suspicion_scores (0-1)
        _, suspicion_scores = model(data)
        
        # Move scores back to CPU and convert to numpy
        scores = suspicion_scores.cpu().numpy()

    # --- 3. Merging with Original Data ---
    print("3. Merging scores with original transactions...")
    
    # Load original data to get account IDs and other details
    df_raw = pd.read_csv(RAW_DATA_PATH)
    df_raw.columns = [col.upper().replace('-', '_') for col in df_raw.columns]
    
    # The scores are ordered exactly as the transactions in the CSV
    df_raw['SUSPICION_SCORE'] = scores
    
    # Select the columns relevant for reporting and visualization
    df_output = df_raw[[
        'SENDER_ACCOUNT_ID', 
        'RECEIVER_ACCOUNT_ID', 
        'TX_AMOUNT', 
        'TX_TYPE',
        'IS_FRAUD', 
        'SUSPICION_SCORE'
    ]].copy()
    
    # --- 4. Saving Results ---
    os.makedirs(os.path.dirname(OUTPUT_SCORES_PATH), exist_ok=True)
    df_output.to_csv(OUTPUT_SCORES_PATH, index=False)
    
    print(f"4. ✅ Suspicion Scores saved to: {OUTPUT_SCORES_PATH}")
    
    # Also print the top 5 highest scored transactions for a quick check
    top_5 = df_output.sort_values(by='SUSPICION_SCORE', ascending=False).head(5)
    print("\nTop 5 Highest Suspicion Scores:")
    print(top_5[['SENDER_ACCOUNT_ID', 'RECEIVER_ACCOUNT_ID', 'TX_AMOUNT', 'IS_FRAUD', 'SUSPICION_SCORE']].to_markdown(index=False))
    
    print("\nPipeline stage 3 COMPLETE. Ready for visualization!")

if __name__ == "__main__":
    if not os.path.exists(MODEL_CHECKPOINT_PATH):
        print(f"Error: Model checkpoint not found at {MODEL_CHECKPOINT_PATH}.")
        print("Please run 'python src/train.py' first.")
    else:
        # Use a safety check to ensure the data is loaded properly before running prediction
        if not os.path.exists(PROCESSED_DATA_PATH):
             print(f"Error: Processed data not found at {PROCESSED_DATA_PATH}.")
             print("Please run 'python src/data_preprocessing.py' first.")
        else:
            generate_predictions()