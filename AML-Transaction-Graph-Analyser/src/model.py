import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class AMLGraphSAGE(torch.nn.Module):
    """
    A Graph Neural Network model (GraphSAGE) designed for Anti-Money Laundering 
    Edge Classification.
    """
    def __init__(self, in_channels_x, in_channels_edge, hidden_channels, out_channels_node):
        super(AMLGraphSAGE, self).__init__()
        
        # 1. Initialize the feature mapper (optional, but good practice)
        # This layer maps the *dummy* node feature (in_channels_x=1) 
        # to the size of the *edge* feature space, allowing for consistent dimensions later.
        self.initial_lin = torch.nn.Linear(in_channels_x, in_channels_edge)
        
        # 2. GraphSAGE Layers
        # We'll use 2 aggregation layers for local neighborhood information flow.
        # Note: The input channel size for SAGEConv must match the size of 
        # the node's feature/embedding vector.
        
        # Layer 1: Input size (in_channels_edge) -> Hidden size (hidden_channels)
        self.conv1 = SAGEConv(in_channels_edge, hidden_channels)
        
        # Layer 2: Hidden size (hidden_channels) -> Output Node size (out_channels_node)
        self.conv2 = SAGEConv(hidden_channels, out_channels_node)

        # 3. Prediction Head (for Edge Classification)
        # It takes the combined embeddings of the sender and receiver nodes 
        # and outputs a single score (the Suspicion Score).
        # Input size is: 2 * out_channels_node (sender + receiver)
        # Output size is: 1 (for binary classification: Fraud/Not-Fraud)
        self.pred_head = torch.nn.Linear(2 * out_channels_node, 1)

    def forward(self, data):
        # Unpack the PyG Data object
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # --- 1. Initial Feature Mapping / Embedding ---
        # Map the dummy node features (x) to a richer initial embedding size
        x = self.initial_lin(x)
        
        # A common practice in PyG when using edge_attr is to pass it 
        # as the 'edge_attr' parameter to the convolution layer, but SAGEConv 
        # in its default form doesn't support edge attributes directly.
        # For simplicity in this base model, we'll leverage the fact that 
        # GraphSAGE will learn embeddings based on the *topology* and the 
        # dummy node features which are then transformed.
        
        # --- 2. GraphSAGE Message Passing ---
        
        # Layer 1
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        
        # Layer 2
        x = self.conv2(x, edge_index)
        
        # At this point, 'x' contains the final, learned embedding vector for EVERY account (node).
        
        # --- 3. Edge Prediction ---
        
        # Get the embeddings for the sender and receiver nodes for ALL transactions (edges)
        sender_embeddings = x[edge_index[0]]
        receiver_embeddings = x[edge_index[1]]
        
        # Concatenate the sender and receiver embeddings (2 * out_channels_node)
        combined_embeddings = torch.cat([sender_embeddings, receiver_embeddings], dim=1)
        
        # Pass through the prediction head to get the raw score (logit)
        raw_score = self.pred_head(combined_embeddings)
        
        # Apply Sigmoid to get the Suspicion Score (0 to 1)
        suspicion_score = torch.sigmoid(raw_score)
        
        # We return the raw score for loss calculation and the suspicion score for output
        return raw_score.squeeze(), suspicion_score.squeeze()
