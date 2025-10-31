import torch
import torch.nn.functional as F
from model_bitcoin import BitcoinGraphSAGE # Import the new model
from sklearn.metrics import f1_score
import os

# --- Configuration (UPDATED PATHS) ---
PROCESSED_DATA_PATH = 'data/processed/graph_data_bitcoin.pt'
MODEL_CHECKPOINT_PATH = 'outputs/model_checkpoint_bitcoin.pth'
NUM_EPOCHS = 200 
LEARNING_RATE = 0.01

# --- Hyperparameters ---
# Load data first to get input dimensions
try:
    data = torch.load(PROCESSED_DATA_PATH, weights_only=False)
except:
    # Exit if data is not loaded (will be handled by the main check)
    exit()

IN_CHANNELS_X = data.x.size(1)              # 166 features
HIDDEN_CHANNELS = 128                      
OUT_CHANNELS_NODE = 64
NUM_CLASSES = 2 

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize the new Bitcoin model
    model = BitcoinGraphSAGE(IN_CHANNELS_X, HIDDEN_CHANNELS, OUT_CHANNELS_NODE).to(device)
    data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # CRITICAL: CrossEntropyLoss for Node Classification (Licit/Illicit)
    # The Elliptic dataset is still highly imbalanced. Calculate pos_weight.
    labeled_nodes = (data.y != -1)
    num_pos = data.y[data.train_mask & labeled_nodes].sum().item()
    num_neg = (data.train_mask & labeled_nodes).sum().item() - num_pos
    
    # Weight = Negatives / Positives
    pos_weight = torch.tensor([num_neg / num_pos], device=device)

    # CrossEntropyLoss requires a class weight tensor for all classes [weight_class_0, weight_class_1]
    # We want class 1 (Illicit) to have the high weight (pos_weight), and class 0 (Licit) to have weight 1.
    class_weights = torch.tensor([1.0, pos_weight.item()], device=device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    best_val_f1 = 0.0
    
    # Filter masks to include only Labeled nodes (Y != -1)
    train_mask = data.train_mask & labeled_nodes
    val_mask = data.val_mask & labeled_nodes
    test_mask = data.test_mask & labeled_nodes
    
    print(f"1. Initializing Model and Data...")
    print(f"   -> Device: {device}")
    print(f"   -> Training nodes (Labeled): {train_mask.sum().item():,}")
    print(f"   -> Positive Weight for Illicit Class: {pos_weight.item():.2f}")
    print(f"\n2. Starting Training Loop (Node Classification)...")

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass returns raw scores (N x 2)
        raw_scores, _ = model(data)
        
        # Calculate loss only on training nodes that are labeled
        loss = criterion(raw_scores[train_mask], data.y[train_mask])
        
        loss.backward()
        optimizer.step()
        
        val_f1 = evaluate(model, data, val_mask)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            os.makedirs(os.path.dirname(MODEL_CHECKPOINT_PATH), exist_ok=True)
            torch.save(model.state_dict(), MODEL_CHECKPOINT_PATH)
            save_status = "-> Model Saved!"
        else:
            save_status = ""
            
        # Evaluation is less frequent to save time
        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | Train Loss: {loss.item():.4f} | Val F1: {val_f1:.4f} {save_status}")
        elif save_status:
            print(f"Epoch {epoch:03d} | Train Loss: {loss.item():.4f} | Val F1: {val_f1:.4f} {save_status}")

    
    # 3. Final Test Evaluation
    model.load_state_dict(torch.load(MODEL_CHECKPOINT_PATH))
    test_f1 = evaluate(model, data, test_mask)
    
    print(f"\nTEST RESULTS: F1 Score: {test_f1:.4f}")
    print("Pipeline stage 2 COMPLETE. Trained Bitcoin model checkpoint saved.")
    
def evaluate(model, data, mask):
    model.eval()
    with torch.no_grad():
        raw_scores, _ = model(data)
        
        # Get predicted class (0 or 1) by finding the max logit
        preds = raw_scores[mask].argmax(dim=1).cpu().numpy()
        labels = data.y[mask].cpu().numpy()
        
        # Use F1-score (binary) for evaluation due to imbalance
        f1_score_val = f1_score(labels, preds, average='binary', zero_division=0)
        return f1_score_val

if __name__ == "__main__":
    if os.path.exists(PROCESSED_DATA_PATH):
        train()
    else:
        print(f"Error: Processed data not found at {PROCESSED_DATA_PATH}. Run data_preprocessing_bitcoin.py first.")
