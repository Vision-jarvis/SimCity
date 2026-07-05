import torch
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def evaluate_model(
    tgn,
    pooling,
    virality_head,
    deffuant,
    influence_scorer,
    hmf_bridge,
    loader,
    neighbor_loader,
    num_nodes,
    num_platforms,
    raw_msg_dim,
    device,
    hawkes_loss_fn=None,
    save_preds_path=None,
):
    """
    Evaluates the SimCity model on a validation/test split.

    If ``save_preds_path`` is given, per-event test predictions are saved as an
    ``.npz`` (dst narrative id, true, pred) so downstream tooling can compute
    within-narrative residual metrics that isolate the temporal/excitation
    signal from a narrative's static reach.
    """
    tgn.eval()
    pooling.eval()
    virality_head.eval()
    deffuant.eval()
    influence_scorer.eval()
    hmf_bridge.eval()
    
    # STRICT GUARD: Enforce memory reset at validation start
    require_reset_check = True
    
    all_preds = []
    all_trues = []
    all_dst = []
    all_intensity = []
    all_plat = []
    all_alpha_off = []
    all_event_plat = []
    all_event_t = []
    hawkes_losses = []
    if hawkes_loss_fn is not None and hasattr(hawkes_loss_fn, "reset_state"):
        hawkes_loss_fn.reset_state()
    influence_scorer.reset_state()
    
    # Deffuant state
    node_opinions = torch.rand(num_nodes, device=device)
    node_exposed = torch.zeros(num_nodes, dtype=torch.long, device=device)
    node_rejected = torch.zeros(num_nodes, dtype=torch.long, device=device)
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            src, pos_dst, t, msg = batch.src, batch.dst, batch.t, batch.msg
            
            n_id, edge_index, e_id = neighbor_loader(torch.cat([src, pos_dst]))
            neighbor_loader.insert(src, pos_dst)
            
            edge_attr = msg[e_id] if e_id is not None and e_id.numel() > 0 and e_id.max() < msg.size(0) else None
            if edge_attr is None:
                edge_attr = torch.zeros((edge_index.size(1), raw_msg_dim), device=device)
            
            # Behavioral Update (Deffuant)
            # Mock content opinion using first feature of msg
            x_c = msg[:, 0]
            
            # Since Deffuant updates continuous state over dt, we pass the time since start of epoch or dt
            dt = torch.ones_like(src, dtype=torch.float32)
            
            x_v_src = node_opinions[src]
            
            x_v_new, rejected, rad = deffuant(
                x_v_src, x_c, dt, 
                influence_scorer.temporal_degree[src].long(),
                (influence_scorer.temporal_degree[src] * 0.5).long(), # Mock cross-community
                node_exposed[src],
                node_rejected[src]
            )
            
            # Update state
            node_opinions[src] = x_v_new
            node_exposed[src] += 1
            node_rejected[src] += rejected.long()
            
            # Full rad vector for TGN forward
            full_rad = torch.zeros(num_nodes, 1, device=device)
            full_rad[src, 0] = rad
            
            # The first batch in validation checks the reset flag.
            h = tgn(n_id, edge_index, edge_attr, src, pos_dst, t, msg, full_rad, require_reset_check=require_reset_check)
            require_reset_check = False # Pass allowed after first check
            
            assoc = torch.empty(num_nodes, dtype=torch.long, device=device)
            assoc[n_id] = torch.arange(n_id.size(0), device=device)
            h_src = h[assoc[src]]
            h_dst_pos = h[assoc[pos_dst]]
            
            platform_ids = batch.platform
            
            # Dynamic Influence Score
            inf_src, inf_dst = influence_scorer(src, pos_dst, t)
            h_P = pooling(h_src, platform_ids, inf_src.detach())
            
            # Update HMF Bridge
            hmf_bridge.update_degree_distribution(influence_scorer.temporal_degree[src], influence_scorer.temporal_degree[pos_dst])
            
            gdelt_volume = msg[:, -1]
            exc_feats = (
                torch.stack([inf_src, inf_dst], dim=-1).detach()
                if getattr(virality_head, "exc_dim", 0) > 0 else None
            )
            beta, mu, alpha, gamma, pred_virality = virality_head(
                h_src, h_dst_pos, h_P, gdelt_volume, exc_feats=exc_feats
            )

            if hawkes_loss_fn is not None:
                if save_preds_path is not None and hasattr(hawkes_loss_fn, "event_intensities"):
                    inten, ord_plat = hawkes_loss_fn.event_intensities(
                        t, platform_ids, mu, alpha, gamma, update_state=False
                    )
                    all_intensity.append(inten.cpu().numpy())
                    all_plat.append(ord_plat.cpu().numpy())
                hawkes_losses.append(
                    hawkes_loss_fn(t, platform_ids, mu, alpha, gamma).detach().cpu().item()
                )
            
            # Mask NaNs for metrics
            mask = ~torch.isnan(batch.y)
            if mask.any():
                all_preds.append(pred_virality[mask].cpu().numpy())
                all_trues.append(batch.y[mask].cpu().numpy())
                all_dst.append(batch.dst[mask].cpu().numpy())
                if save_preds_path is not None:
                    # Per-event cross-platform excitation mass (off-diagonal
                    # alpha / decay-normalised): the narrative-conditioned
                    # "bridge" signal used by the transfer-detection metric.
                    P_ = alpha.size(-1)
                    off = ~torch.eye(P_, dtype=torch.bool, device=alpha.device)
                    branch = alpha / gamma.clamp_min(1e-6)
                    all_alpha_off.append(
                        branch[mask][:, off].mean(dim=-1).detach().cpu().numpy()
                    )
                    all_event_plat.append(batch.platform[mask].cpu().numpy())
                    all_event_t.append(batch.t[mask].cpu().numpy())
                
    metrics = {"virality_mae": None, "virality_rmse": None, "hawkes_nll": None}
    if len(all_preds) > 0:
        preds = np.concatenate(all_preds)
        trues = np.concatenate(all_trues)
        mae = mean_absolute_error(trues, preds)
        rmse = np.sqrt(mean_squared_error(trues, preds))
        metrics["virality_mae"] = mae
        metrics["virality_rmse"] = rmse
        if save_preds_path is not None:
            dst = np.concatenate(all_dst)
            extra = {}
            if all_alpha_off:
                extra = {
                    "alpha_off": np.concatenate(all_alpha_off),
                    "event_platform": np.concatenate(all_event_plat),
                    "event_t": np.concatenate(all_event_t),
                }
            np.savez(save_preds_path, dst=dst, true=trues, pred=preds, **extra)
            if all_intensity:
                inten_path = save_preds_path.replace(".npz", "") + "_hawkes.npz"
                np.savez(
                    inten_path,
                    intensity=np.concatenate(all_intensity),
                    platform=np.concatenate(all_plat),
                )
        msg = f"Evaluation -> Virality MAE: {mae:.4f} | RMSE: {rmse:.4f}"
        if hawkes_losses:
            metrics["hawkes_nll"] = float(np.mean(hawkes_losses))
            msg += f" | Hawkes NLL: {metrics['hawkes_nll']:.4f}"
        print(msg)
    else:
        print("Evaluation -> No valid virality targets found.")
        
    return metrics
