import torch
from torch import nn
from torch_geometric.nn.models.tgn import (
    IdentityMessage,
    LastAggregator,
    TGNMemory
)
from torch_geometric.nn import TransformerConv

class GraphAttentionEmbedding(nn.Module):
    def __init__(self, in_channels, out_channels, msg_dim, time_enc):
        super().__init__()
        self.time_enc = time_enc
        edge_dim = msg_dim + time_enc.out_channels
        self.conv = TransformerConv(in_channels, out_channels // 4, heads=4,
                                    dropout=0.2, edge_dim=edge_dim)

    def forward(self, x, last_update, edge_index, msg):
        rel_t = last_update[edge_index[0]] - last_update[edge_index[1]]
        rel_t_enc = self.time_enc(rel_t.to(x.dtype).clamp(min=0))
        edge_attr = torch.cat([rel_t_enc, msg], dim=-1)
        return self.conv(x, edge_index, edge_attr)

class SimCityTGN(torch.nn.Module):
    """
    Core Temporal Graph Network for SimCity.
    Maintains continuous-time memory states for users and narratives.
    """
    def __init__(self, num_nodes, raw_msg_dim, memory_dim, time_dim, embedding_dim):
        super().__init__()
        
        self.memory = TGNMemory(
            num_nodes=num_nodes,
            raw_msg_dim=raw_msg_dim,
            memory_dim=memory_dim,
            time_dim=time_dim,
            message_module=IdentityMessage(raw_msg_dim, memory_dim, time_dim),
            aggregator_module=LastAggregator(),
        )
        # Override PyG's default torch.long buffer to support float timestamps
        self.memory.register_buffer('last_update', torch.empty(num_nodes, dtype=torch.float))
        
        # Monkey-patch _reset_message_store to use float for empty timestamp tensors
        def _custom_reset_message_store():
            i = self.memory.memory.new_empty((0, ), device=self.memory.device, dtype=torch.long)
            t_empty = self.memory.memory.new_empty((0, ), device=self.memory.device, dtype=torch.float)
            msg = self.memory.memory.new_empty((0, self.memory.raw_msg_dim), device=self.memory.device)
            self.memory.msg_s_store = {j: (i, i, t_empty, msg) for j in range(self.memory.num_nodes)}
            self.memory.msg_d_store = {j: (i, i, t_empty, msg) for j in range(self.memory.num_nodes)}
            
        self.memory._reset_message_store = _custom_reset_message_store
        self.memory.reset_state()
        
        self.gat_embedding = GraphAttentionEmbedding(
            in_channels=memory_dim + 1,
            out_channels=embedding_dim,
            msg_dim=raw_msg_dim,
            time_enc=self.memory.time_enc,
        )
        
        # Point-wise MLP for negative samples (memory only, no neighborhood)
        self.embedding_mlp = nn.Sequential(
            nn.Linear(memory_dim + 1, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
        self._memory_was_reset = True

    def forward(self, n_id, edge_index, edge_attr, src, dst, t, msg, rad, require_reset_check=False):
        """
        Updates memory and calculates the temporal node embeddings.
        """
        if require_reset_check and not self._memory_was_reset:
            raise RuntimeError(
                "Memory was not reset before this forward pass. "
                "Call reset_memory() at split boundaries."
            )
        self._memory_was_reset = False

        # Fetch pre-update memory for all nodes in computational graph
        z, last_update = self.memory(n_id)
        
        # Concatenate behavioral radicalization scalar with structural memory
        # rad is expected to be shape (num_nodes, 1)
        z_with_rad = torch.cat([z, rad[n_id]], dim=-1)
        
        # Compute embeddings via temporal attention over neighborhood
        h = self.gat_embedding(z_with_rad, last_update, edge_index, edge_attr)
        
        # Update memory with the new batch events
        self.memory.update_state(src, dst, t, msg)
        
        return h

    def reset_memory(self):
        """
        Resets the TGN memory state. MUST be called at the train/val/test boundary
        to strictly prevent temporal data leakage.
        """
        self.memory.reset_state()
        self._memory_was_reset = True
