import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class StaticGNN_SEIR(nn.Module):
    """
    Static GNN + SEIR Baseline.
    Aggregates temporal edges into a static snapshot and runs GraphSAGE,
    then predicts SEIR parameters using MLPs.
    Ablates the continuous-time and stochastic components of SimCity.
    """
    def __init__(self, num_nodes, in_channels, hidden_channels, out_channels):
        super().__init__()
        # Node features
        self.node_emb = nn.Embedding(num_nodes, in_channels)
        
        # GraphSAGE layers
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
        
        # SEIR parameter MLPs
        self.mlp_beta = nn.Sequential(
            nn.Linear(out_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1)
        )
        
    def forward(self, edge_index, src, dst):
        """
        IMPORTANT: edge_index passed to forward() must be pre-filtered to contain
        only edges with timestamp t <= current query time. Failure to do so
        constitutes temporal leakage and invalidates baseline comparisons.
        
        Args:
            edge_index: (2, E) all edges up to current snapshot
            src: (B,) source nodes to predict
            dst: (B,) destination nodes to predict
        Returns:
            beta: (B,) predicted infectivity rate
            h_src: (B, D) source embeddings
            h_dst: (B, D) destination embeddings
        """
        # Static embeddings
        x = self.node_emb.weight
        
        # GNN passes
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        
        # Extract embeddings for the queried pairs
        h_src = x[src]
        h_dst = x[dst]
        
        z = torch.cat([h_src, h_dst], dim=-1)
        
        beta = F.softplus(self.mlp_beta(z)).squeeze(-1)
        
        return beta, h_src, h_dst
