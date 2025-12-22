import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from sklearn.metrics import roc_auc_score, f1_score
import numpy as np


from model import AMLGraphSAGE # Import the model you defined

import torch_geometric.data.data
torch.serialization.add_safe_globals([torch_geometric.data.data.Data])

# --- Configuration ---
PROCESSED_DATA_PATH = 'data/processed/graph_data.pt'
MODEL_CHECKPOINT_PATH = 'outputs/model_checkpoint.pth'
NUM_EPOCHS = 100 # Keep low for quick demo, increase later
LEARNING_RATE = 0.01

# --- Hyperparameters for the Model ---
# Load data first to get input dimensions
data = torch.load(PROCESSED_DATA_PATH, weights_only=False)
IN_CHANNELS_X = data.x.size(1)              # Should be 1 (dummy feature)
IN_CHANNELS_EDGE = data.edge_attr.size(1)   # Should be 2 (Amount + TX_Types)
HIDDEN_CHANNELS = 64
OUT_CHANNELS_NODE = 32

def train():
    """Implements the training and validation loop."""
    print("1. Initializing Model and Data...")
    
    # Initialize the model, optimizer, and criterion
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AMLGraphSAGE(IN_CHANNELS_X, IN_CHANNELS_EDGE, HIDDEN_CHANNELS, OUT_CHANNELS_NODE).to(device)
    data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- CRITICAL: Handle Class Imbalance ---
    # Calculate the positive and negative weights for the Binary Cross-Entropy Loss.
    # We want to heavily penalize False Negatives (missing fraud).
    num_pos = data.y[data.train_mask].sum().item()
    num_neg = data.train_mask.sum().item() - num_pos
    
    # Weight = Negatives / Positives
    pos_weight = torch.tensor([num_neg / num_pos], device=device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    print(f"   -> Device: {device}")
    print(f"   -> Training edges (Fraud/Total): {num_pos}/{data.train_mask.sum().item()}")
    print(f"   -> Positive Weight for BCE Loss: {pos_weight.item():.2f}")
    
    best_val_auc = 0.0
    
    # --- 2. Training Loop ---
    print("\n2. Starting Training Loop...")
    for epoch in range(1, NUM_EPOCHS + 1):
        # 2a. Training Step
        model.train()
        optimizer.zero_grad()
        
        # Forward pass: model returns (raw_scores, suspicion_scores)
        raw_scores, _ = model(data)
        
        # Calculate loss only on training edges
        loss = criterion(raw_scores[data.train_mask], data.y[data.train_mask])
        
        loss.backward()
        optimizer.step()
        
        # 2b. Evaluation (Validation Step)
        val_loss, val_metrics = evaluate(model, data, data.val_mask, criterion)

        if val_metrics['roc_auc'] > best_val_auc:
            best_val_auc = val_metrics['roc_auc']
            # Save the best model
            os.makedirs(os.path.dirname(MODEL_CHECKPOINT_PATH), exist_ok=True)
            torch.save(model.state_dict(), MODEL_CHECKPOINT_PATH)
            save_status = "-> Model Saved!"
        else:
            save_status = ""
            
        print(f"Epoch {epoch:03d} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f} | Val AUC: {val_metrics['roc_auc']:.4f} | Val F1: {val_metrics['f1']:.4f} {save_status}")
    
    # --- 3. Final Test Evaluation ---
    print("\n3. Final Test Evaluation...")
    # Load the best model weights
    model.load_state_dict(torch.load(MODEL_CHECKPOINT_PATH))
    test_loss, test_metrics = evaluate(model, data, data.test_mask, criterion)
    
    print(f"TEST RESULTS: Loss: {test_loss:.4f} | ROC-AUC: {test_metrics['roc_auc']:.4f} | F1 Score: {test_metrics['f1']:.4f}")
    
    print("\nPipeline stage 2 COMPLETE. Trained model checkpoint saved.")
    
def evaluate(model, data, mask, criterion):
    """Evaluates the model on a given mask (val or test)."""
    model.eval()
    with torch.no_grad():
        raw_scores, suspicion_scores = model(data)
        
        # Loss calculation
        loss = criterion(raw_scores[mask], data.y[mask])

        # Metric calculation
        preds = suspicion_scores[mask].cpu().numpy()
        labels = data.y[mask].cpu().numpy()
        
        try:
            auc = roc_auc_score(labels, preds)
        except ValueError:
            # Handle case where only one class is present in the batch
            auc = 0.0
            
        # F1 Score requires converting scores to binary predictions (0 or 1)
        # We use a threshold of 0.5 for simplicity, but a better threshold 
        # could be learned on the validation set.
        binary_preds = (preds > 0.5).astype(int)
        f1 = f1_score(labels, binary_preds, zero_division=0)
        
        metrics = {'roc_auc': auc, 'f1': f1}
        return loss.item(), metrics

if __name__ == "__main__":
    if not os.path.exists(PROCESSED_DATA_PATH):
        print(f"Error: Processed data not found at {PROCESSED_DATA_PATH}.")
        print("Please run 'python src/data_preprocessing.py' first.")
    else:
        train()
