import torch
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def evaluate_model(tgn, pooling, virality_head, loader, neighbor_loader, num_nodes, num_platforms, raw_msg_dim, device):
    """
    Evaluates the SimCity model on a validation/test split.
    """
    tgn.eval()
    pooling.eval()
    virality_head.eval()
    
    # STRICT GUARD: Enforce memory reset at validation start
    require_reset_check = True
    
    all_preds = []
    all_trues = []
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            src, pos_dst, t, msg = batch.src, batch.dst, batch.t, batch.msg
            
            n_id, edge_index, e_id = neighbor_loader(torch.cat([src, pos_dst]))
            neighbor_loader.insert(src, pos_dst)
            
            edge_attr = msg[e_id] if e_id is not None and e_id.numel() > 0 and e_id.max() < msg.size(0) else None
            if edge_attr is None:
                edge_attr = torch.zeros((edge_index.size(1), raw_msg_dim), device=device)
            
            # The first batch in validation checks the reset flag.
            h = tgn(n_id, edge_index, edge_attr, src, pos_dst, t, msg, require_reset_check=require_reset_check)
            require_reset_check = False # Pass allowed after first check
            
            assoc = torch.empty(num_nodes, dtype=torch.long, device=device)
            assoc[n_id] = torch.arange(n_id.size(0), device=device)
            h_src = h[assoc[src]]
            h_dst_pos = h[assoc[pos_dst]]
            
            platform_ids = batch.platform
            influence_scores = h_src.norm(dim=-1).detach()
            h_P = pooling(h_src, platform_ids, influence_scores)
            
            beta, alpha, gamma, pred_virality = virality_head(h_src, h_dst_pos, h_P)
            
            # Mask NaNs for metrics
            mask = ~torch.isnan(batch.y)
            if mask.any():
                all_preds.append(pred_virality[mask].cpu().numpy())
                all_trues.append(batch.y[mask].cpu().numpy())
                
    mae, rmse = None, None
    if len(all_preds) > 0:
        preds = np.concatenate(all_preds)
        trues = np.concatenate(all_trues)
        mae = mean_absolute_error(trues, preds)
        rmse = np.sqrt(mean_squared_error(trues, preds))
        print(f"Evaluation -> Virality MAE: {mae:.4f} | RMSE: {rmse:.4f}")
    else:
        print("Evaluation -> No valid virality targets found.")
        
    return mae, rmse
