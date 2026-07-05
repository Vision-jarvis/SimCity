"""Train the two graph baselines for the virality-regression table.

Both baselines consume the *same* chronological 70/10/20 split and the same
log-engagement target as the full SimCity model, so their MAE/RMSE are directly
comparable to ``results/simcity_metrics.json``.

  * Vanilla TGN   -- SimCity's continuous-time TGN core with a plain virality
                     readout, but WITHOUT the neural-Hawkes head, HMF bridge,
                     Deffuant behaviour, influence pooling, or radicalization
                     coupling (rad is held at 0). Isolates what the
                     epidemiological machinery adds over a standard TGN.
  * Static GNN    -- GraphSAGE over a single static snapshot built only from
                     TRAIN edges (no future edges => no temporal leakage),
                     with the same virality readout. Ablates continuous time.

Usage:
    python -m baselines.run_virality_baselines --epochs 15
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import TemporalDataLoader
from torch_geometric.nn import SAGEConv
from torch_geometric.nn.models.tgn import LastNeighborLoader

from data.dataset import load_simcity_temporal_data
from models.tgn_core import SimCityTGN


def _mae_rmse(preds, trues):
    preds = np.asarray(preds, dtype=np.float64)
    trues = np.asarray(trues, dtype=np.float64)
    mae = float(np.abs(preds - trues).mean())
    rmse = float(np.sqrt(((preds - trues) ** 2).mean()))
    return mae, rmse


# --------------------------------------------------------------------------
# Baseline 1: Vanilla TGN + virality readout
# --------------------------------------------------------------------------
def train_vanilla_tgn(data, train_idx, val_idx, test_idx, epochs, device):
    train_data = data[train_idx]
    test_data = data[test_idx]
    num_nodes = data.num_nodes
    raw_msg_dim = data.msg.size(-1)
    emb_dim = 256

    tgn = SimCityTGN(num_nodes, raw_msg_dim, emb_dim, emb_dim, emb_dim).to(device)
    readout = nn.Sequential(
        nn.Linear(2 * emb_dim, 64), nn.ReLU(), nn.Linear(64, 1)
    ).to(device)

    valid_y = train_data.y[~torch.isnan(train_data.y)]
    tgt_mean = valid_y.mean().item() if valid_y.numel() else 0.0
    tgt_std = valid_y.std(unbiased=False).clamp_min(1e-6).item() if valid_y.numel() else 1.0

    opt = torch.optim.AdamW(
        list(tgn.parameters()) + list(readout.parameters()), lr=1e-3, weight_decay=1e-4
    )
    neighbor_loader = LastNeighborLoader(num_nodes, size=10, device=device)
    zero_rad = torch.zeros(num_nodes, 1, device=device)

    def run_split(loader, train_mode, collect_dst=False):
        tgn.train(train_mode)
        readout.train(train_mode)
        preds, trues, dsts, tot = [], [], [], 0.0
        for batch in loader:
            batch = batch.to(device)
            src, dst, t, msg = batch.src, batch.dst, batch.t, batch.msg
            n_id, edge_index, e_id = neighbor_loader(torch.cat([src, dst]))
            neighbor_loader.insert(src, dst)
            edge_attr = (
                msg[e_id]
                if e_id is not None and e_id.numel() > 0 and e_id.max() < msg.size(0)
                else torch.zeros((edge_index.size(1), raw_msg_dim), device=device)
            )
            if train_mode:
                h = tgn(n_id, edge_index, edge_attr, src, dst, t, msg, zero_rad)
            else:
                with torch.no_grad():
                    h = tgn(n_id, edge_index, edge_attr, src, dst, t, msg, zero_rad)
            assoc = torch.empty(num_nodes, dtype=torch.long, device=device)
            assoc[n_id] = torch.arange(n_id.size(0), device=device)
            z = torch.cat([h[assoc[src]], h[assoc[dst]]], dim=-1)
            pred = readout(z).squeeze(-1) * tgt_std + tgt_mean
            mask = ~torch.isnan(batch.y)
            if mask.any():
                loss = F.mse_loss(pred[mask], batch.y[mask])
                if train_mode:
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    tgn.memory.detach()
                preds.append(pred[mask].detach().cpu().numpy())
                trues.append(batch.y[mask].detach().cpu().numpy())
                if collect_dst:
                    dsts.append(batch.dst[mask].detach().cpu().numpy())
                tot += loss.item()
        if collect_dst:
            return preds, trues, dsts
        return preds, trues

    for ep in range(epochs):
        tgn.reset_memory()
        neighbor_loader.reset_state()
        run_split(TemporalDataLoader(train_data, batch_size=200), train_mode=True)

    tgn.reset_memory()
    neighbor_loader.reset_state()
    # warm memory over train, then evaluate test without leakage
    run_split(TemporalDataLoader(train_data, batch_size=200), train_mode=False)
    preds, trues, dsts = run_split(
        TemporalDataLoader(test_data, batch_size=200), train_mode=False, collect_dst=True
    )
    p, tr = np.concatenate(preds), np.concatenate(trues)
    mae, rmse = _mae_rmse(p, tr)
    np.savez("results/vanilla_tgn_test_preds.npz",
             dst=np.concatenate(dsts), true=tr, pred=p)
    return {"virality_mae": mae, "virality_rmse": rmse}


# --------------------------------------------------------------------------
# Baseline 2: Static GraphSAGE snapshot + virality readout
# --------------------------------------------------------------------------
class StaticGNNVirality(nn.Module):
    def __init__(self, num_nodes, in_dim=128, hid=128, out=128):
        super().__init__()
        self.emb = nn.Embedding(num_nodes, in_dim)
        self.conv1 = SAGEConv(in_dim, hid)
        self.conv2 = SAGEConv(hid, out)
        self.readout = nn.Sequential(nn.Linear(2 * out, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, edge_index, src, dst):
        x = F.relu(self.conv1(self.emb.weight, edge_index))
        x = self.conv2(x, edge_index)
        return self.readout(torch.cat([x[src], x[dst]], dim=-1)).squeeze(-1)


def train_static_gnn(data, train_idx, val_idx, test_idx, epochs, device):
    num_nodes = data.num_nodes
    # Snapshot graph built ONLY from training edges (no future leakage).
    tr_src = data.src[train_idx].to(device)
    tr_dst = data.dst[train_idx].to(device)
    edge_index = torch.stack([tr_src, tr_dst], dim=0)
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)  # undirected

    model = StaticGNNVirality(num_nodes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-3, weight_decay=1e-4)

    y = data.y.to(device)
    tr_y = y[train_idx]
    tr_mask = ~torch.isnan(tr_y)
    tgt_mean = tr_y[tr_mask].mean().item()
    tgt_std = tr_y[tr_mask].std(unbiased=False).clamp_min(1e-6).item()

    s_tr, d_tr, yy_tr = tr_src[tr_mask], tr_dst[tr_mask], tr_y[tr_mask]
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        pred = model(edge_index, s_tr, d_tr) * tgt_std + tgt_mean
        loss = F.mse_loss(pred, yy_tr)
        loss.backward()
        opt.step()

    model.eval()
    te_src = data.src[test_idx].to(device)
    te_dst = data.dst[test_idx].to(device)
    te_y = y[test_idx]
    te_mask = ~torch.isnan(te_y)
    with torch.no_grad():
        pred = model(edge_index, te_src[te_mask], te_dst[te_mask]) * tgt_std + tgt_mean
    p = pred.cpu().numpy()
    tr = te_y[te_mask].cpu().numpy()
    mae, rmse = _mae_rmse(p, tr)
    os.makedirs("results", exist_ok=True)
    np.savez("results/static_gnn_test_preds.npz",
             dst=te_dst[te_mask].cpu().numpy(), true=tr, pred=p)
    return {"virality_mae": mae, "virality_rmse": rmse}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/synthetic_events.pkl")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()

    device = torch.device("cpu")
    data, train_idx, val_idx, test_idx = load_simcity_temporal_data(args.data)

    print("Training Static GNN + SEIR baseline...")
    static_res = train_static_gnn(data, train_idx, val_idx, test_idx, args.epochs, device)
    print(f"  Static GNN test: {static_res}")

    print("Training Vanilla TGN baseline...")
    tgn_res = train_vanilla_tgn(data, train_idx, val_idx, test_idx, args.epochs, device)
    print(f"  Vanilla TGN test: {tgn_res}")

    os.makedirs(args.out_dir, exist_ok=True)
    out = {
        "epochs": args.epochs,
        "static_gnn_seir": static_res,
        "vanilla_tgn": tgn_res,
    }
    path = os.path.join(args.out_dir, "graph_baselines.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
