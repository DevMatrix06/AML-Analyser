import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class BitcoinGraphSAGE(torch.nn.Module):
    """
    GraphSAGE model tailored for the Elliptic Bitcoin dataset.
    Performs Node Classification (predicting Licit/Illicit for each transaction address).
    """
    def __init__(self, in_channels_x, hidden_channels, out_channels_node):
        super(BitcoinGraphSAGE, self).__init__()
        
        # 1. GraphSAGE Layers
        # Input: 166 features (in_channels_x)
        # We skip the initial_lin layer since input features are rich
        self.conv1 = SAGEConv(in_channels_x, hidden_channels)
        
        # Layer 2
        self.conv2 = SAGEConv(hidden_channels, out_channels_node)

        # 2. Prediction Head (maps node embedding to 2 output classes)
        # Output size is 2: Class 0 (Licit) and Class 1 (Illicit)
        self.pred_head = torch.nn.Linear(out_channels_node, 2)

    def forward(self, data):
        # Unpack the PyG Data object (Elliptic data only has x and edge_index)
        x, edge_index = data.x, data.edge_index
        
        # --- 1. GraphSAGE Message Passing ---
        
        # Layer 1
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        
        # Layer 2
        x = self.conv2(x, edge_index)
        
        # --- 2. Node Prediction ---
        
        # raw_scores shape: N_NODES x 2 (logits for CrossEntropyLoss)
        raw_scores = self.pred_head(x)
        
        # suspicion_scores: Probability of being Illicit (Class 1)
        suspicion_scores = F.softmax(raw_scores, dim=1)[:, 1]
        
        # We return both raw logits (for loss) and the illicit probability score
        return raw_scores, suspicion_scores
