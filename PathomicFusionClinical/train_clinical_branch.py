import os
import re
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split, StratifiedKFold
import argparse
import csv
from typing import Optional, Tuple, Dict, List

CLINICAL_MAP_PKL = "data/TCGA_GBMLGG/clinical_map.pkl"
SURVIVAL_LABELS_PKL = "data/TCGA_GBMLGG/survival_labels.pkl"
PNAS_SPLITS_CSV = "data/TCGA_GBMLGG/pnas_splits.csv"
OUT_DIR = "./model_checkpoints/clinical_branch"
os.makedirs(OUT_DIR, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def extract_mappings(clinical_map_obj):
    """
    Expect clinical_map_obj to be a dict with key 'mappings' or 'mapping' mapping TCGA ID -> embedding
    """
    if isinstance(clinical_map_obj, dict):
        if "mapping" in clinical_map_obj:
            mappings = clinical_map_obj["mapping"]
            return mappings
    raise RuntimeError("clinical_map.pkl structure not recognized. Expected dict with key 'mappings'/'mapping' or a dict mapping ids->embeddings.")

def infer_survival_dict(surv_obj):
    """
    Attempt to coerce survival_labels.pkl into dict: tcga_id -> {"time":float, "event":0/1}
    """
    if isinstance(surv_obj, dict):
        sample_keys = list(surv_obj.keys())[:5]
        sample_val = surv_obj[sample_keys[0]] if sample_keys else None
        # case: id -> (time, event)
        if isinstance(sample_val, (tuple, list)) and len(sample_val) >= 2:
            out = {}
            for k, v in surv_obj.items():
                t = float(v[0])
                e = int(v[1])
                out[str(k)] = {"time": t, "event": e}
            return out
        # case: id -> dict with time/event-like keys
        if isinstance(sample_val, dict):
            out = {}
            for k, v in surv_obj.items():
                # tolerant names
                time_keys = [kk for kk in v.keys() if "time" in kk.lower() or "month" in kk.lower() or "survival" in kk.lower()]
                event_keys = [kk for kk in v.keys() if "event" in kk.lower() or "status" in kk.lower() or "dead" in kk.lower()]
                if time_keys:
                    t = float(v[time_keys[0]])
                else:
                    vals = [x for x in v.values() if (isinstance(x, (int, float)))]
                    t = float(vals[0]) if vals else np.nan
                if event_keys:
                    e = int(v[event_keys[0]])
                else:
                    e = int(v.get("event", v.get("status", 0)))
                out[str(k)] = {"time": t, "event": e}
            return out
        # unknown dict layout
        raise RuntimeError("survival_labels.pkl is dict but values not recognized. Sample value: {}".format(sample_val))

    # If it's a pandas-like object or dict-of-columns, coerce to DF
    try:
        df = pd.DataFrame(surv_obj)
        # if columns include sample id
        if "sample_id" in df.columns:
            id_col = "sample_id"
        else:
            id_col = None
        time_cols = [c for c in df.columns if "time" in c.lower() or "month" in c.lower() or "survival" in c.lower()]
        event_cols = [c for c in df.columns if "event" in c.lower() or "status" in c.lower() or "dead" in c.lower()]
        if len(time_cols) == 0 or len(event_cols) == 0:
            numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            if len(numeric) >= 2:
                time_cols = [numeric[0]]
                event_cols = [numeric[1]]
            else:
                raise RuntimeError("Could not find time/event columns in survival_labels.pkl DataFrame.")
        out = {}
        for idx, row in df.iterrows():
            key = row[id_col] if id_col else idx
            out[str(key)] = {"time": float(row[time_cols[0]]), "event": int(row[event_cols[0]])}
        return out
    except Exception:
        raise RuntimeError("Unable to interpret survival_labels.pkl format. Please inspect file manually.")

def build_dataset_from_pickles(clin_map_p, surv_p):
    print("Loading clinical map:", clin_map_p)
    clinical_map_obj = load_pickle(clin_map_p)
    print("Loading survival labels:", surv_p)
    surv_obj = load_pickle(surv_p)

    mappings_raw = extract_mappings(clinical_map_obj)
    surv_dict_raw = infer_survival_dict(surv_obj)

    # Normalize mapping keys to strings (handles bytes vs str)
    mappings = {str(k): np.array(v, dtype=np.float32) for k, v in mappings_raw.items()}
    surv_dict = {str(k): v for k, v in surv_dict_raw.items()}

    map_ids = set(mappings.keys())
    surv_ids = set(surv_dict.keys())
    common = sorted(map_ids & surv_ids)
    print(f"{len(map_ids)} embeddings, {len(surv_ids)} survival entries, {len(common)} common IDs")

    if len(common) == 0:
        raise RuntimeError("No overlapping TCGA IDs between clinical_map and survival_labels!")

    embeddings = []
    times = []
    events = []
    ids = []
    for tid in common:
        emb = mappings.get(tid)
        if emb is None:
            print("Skipping ID (no embedding found):", tid)
            continue
        s = surv_dict[tid]
        t = float(s["time"])
        e = int(s["event"])
        embeddings.append(np.array(emb, dtype=np.float32))
        times.append(t)
        events.append(e)
        ids.append(tid)
    embeddings = np.vstack(embeddings)
    times = np.array(times, dtype=np.float32)
    events = np.array(events, dtype=np.int32)
    print("Built dataset: N =", embeddings.shape[0], "D =", embeddings.shape[1])
    return embeddings, times, events, ids

class ClinicalSurvDataset(Dataset):
    def __init__(self, embeddings, times, events, ids=None):
        self.emb = torch.tensor(embeddings, dtype=torch.float32)
        self.times = torch.tensor(times, dtype=torch.float32)
        self.events = torch.tensor(events, dtype=torch.float32)
        self.ids = ids

    def __len__(self):
        return len(self.emb)

    def __getitem__(self, idx):
        return {"emb": self.emb[idx], "time": self.times[idx], "event": self.events[idx], "id": (self.ids[idx] if self.ids else idx)}

class ClinicalNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, output_dim: int = 32):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),  # Wider layer
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(hidden_dim, output_dim),
            nn.ReLU()
        )
        self.out_dim = output_dim   # <-- important so script can infer feature dim
        print(f"ClinicalNet: {input_dim} → {hidden_dim} → {output_dim}")

    def forward(self, x):
        return self.network(x)

class ClinicalSurvivalModel(nn.Module):
    def __init__(self, clinical_network: nn.Module, feature_dim: int):
        super().__init__()
        self.backbone = clinical_network
        self.head = nn.Linear(feature_dim, 1)

    def forward(self, x):
        feats = self.backbone(x)
        if feats.dim() == 1:
            feats = feats.unsqueeze(0)
        out = self.head(feats).squeeze(-1)
        return out

def logcumsumexp_manual(x: torch.Tensor) -> torch.Tensor:
    # gradient-safe manual version using cummax (no in-place writes)
    xmax_vals, _ = torch.cummax(x, dim=0)   # returns (values, indices)
    y = x - xmax_vals
    yexp = torch.exp(y)
    yexp_cumsum = torch.cumsum(yexp, dim=0)
    return xmax_vals + torch.log(yexp_cumsum + 1e-12)

def neg_cox_ph_loss_full(risk_scores: torch.Tensor, times: torch.Tensor, events: torch.Tensor) -> torch.Tensor:
    device = risk_scores.device
    times = times.to(device)
    events = events.to(device)

    order = torch.argsort(times, descending=True)
    rs = risk_scores[order]
    ev = events[order]

    # prefer builtin if available
    if hasattr(torch, "logcumsumexp"):
        lse = torch.logcumsumexp(rs, dim=0)
    else:
        lse = logcumsumexp_manual(rs)

    diff = rs - lse
    ll = (diff * ev).sum()
    neg_ll = -ll / (ev.sum() + 1e-12)
    return neg_ll

def concordance_index(event_times: np.ndarray, predicted_scores: np.ndarray, events: np.ndarray) -> float:
    n = 0
    concordant = 0.0
    for i in range(len(event_times)):
        for j in range(i + 1, len(event_times)):
            ti, tj = event_times[i], event_times[j]
            ei, ej = events[i], events[j]
            si, sj = predicted_scores[i], predicted_scores[j]
            # comparable if one had event and earlier time
            if (ei == 1 and ti < tj) or (ej == 1 and tj < ti):
                n += 1
                if ti < tj:
                    if si > sj:
                        concordant += 1
                    elif si == sj:
                        concordant += 0.5
                else:
                    if sj > si:
                        concordant += 1
                    elif sj == si:
                        concordant += 0.5
    if n == 0:
        return float('nan')
    return concordant / n

def train_epoch(model, optimizer, dataset: ClinicalSurvDataset, device):
    model.train()
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    total_loss = 0.0
    steps = 0
    for batch in loader:
        emb = batch["emb"].to(device)
        times = batch["time"].to(device)
        events = batch["event"].to(device)
        scores = model(emb)
        # batch-approx Cox update (common practice). Exact full-dataset gradient steps would be slower.
        loss = neg_cox_ph_loss_full(scores, times, events)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())
        steps += 1
    return total_loss / max(1, steps)

def evaluate_full(model, embeddings_np, times_np, events_np, device):
    model.eval()
    with torch.no_grad():
        emb = torch.tensor(embeddings_np, dtype=torch.float32).to(device)
        scores = model(emb).cpu().numpy()
    cidx = concordance_index(times_np, scores, events_np)
    return {"c_index": cidx, "scores": scores}

def load_pnas_splits():
    df = pd.read_csv(PNAS_SPLITS_CSV, dtype=str)
    df = df.rename(columns={df.columns[0]: "TCGA_ID"})
    df = df.set_index("TCGA_ID")
    return df

def get_fold_split(split_df, fold_idx):
    col = f"Randomization - {fold_idx}"
    train_ids = split_df.index[split_df[col] == "Train"].tolist()
    test_ids = split_df.index[split_df[col] == "Test"].tolist()
    return train_ids, test_ids

def main():
    emb, times, events, ids = build_dataset_from_pickles(CLINICAL_MAP_PKL, SURVIVAL_LABELS_PKL)
    id_to_idx = {tid: i for i, tid in enumerate(ids)}

    split_df = load_pnas_splits()
    fold_results = []

    for fold in range(1, 16):
        print(f"\n========== FOLD {fold} / 15 ==========")
        train_ids, test_ids = get_fold_split(split_df, fold)
        train_idx = [id_to_idx[i] for i in train_ids if i in id_to_idx]
        test_idx = [id_to_idx[i] for i in test_ids if i in id_to_idx]

        # Model setup
        in_dim = emb.shape[1]
        clinical_net = ClinicalNet(input_dim=in_dim, hidden_dim=256, output_dim=32)
        feat_dim = 32
        model = ClinicalSurvivalModel(clinical_net, feature_dim=feat_dim).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

        # Datasets
        train_ds = ClinicalSurvDataset(emb[train_idx], times[train_idx], events[train_idx])
        test_ds = ClinicalSurvDataset(emb[test_idx], times[test_idx], events[test_idx])

        # Prepare per-fold metrics CSV
        fold_metrics_csv = os.path.join(OUT_DIR, f"fold_{fold}_metrics.csv")
        with open(fold_metrics_csv, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["epoch", "train_loss", "train_cindex", "test_loss", "test_cindex"])

        # 10 epochs per fold
        for epoch in range(30):
            train_loss = train_epoch(model, optimizer, train_ds, DEVICE)

            train_metrics = evaluate_full(model, emb[train_idx], times[train_idx], events[train_idx], DEVICE)
            test_metrics = evaluate_full(model, emb[test_idx], times[test_idx], events[test_idx], DEVICE)

            train_cidx = train_metrics["c_index"]
            test_cidx = test_metrics["c_index"]

            with torch.no_grad():
                emb_test = torch.tensor(emb[test_idx], dtype=torch.float32).to(DEVICE)
                times_test = torch.tensor(times[test_idx], dtype=torch.float32).to(DEVICE)
                events_test = torch.tensor(events[test_idx], dtype=torch.float32).to(DEVICE)
                test_scores = model(emb_test)
                test_loss = neg_cox_ph_loss_full(test_scores, times_test, events_test).item()

            with open(fold_metrics_csv, "a", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([epoch, train_loss, train_cidx, test_loss, test_cidx])

            print(f"Fold {fold} Epoch {epoch} | train_loss {train_loss:.4f} | train_cidx {train_cidx:.4f} "
                  f"| test_loss {test_loss:.4f} | test_cidx {test_cidx:.4f}")

        # Save model weights for this fold
        fold_model_path = os.path.join(OUT_DIR, f"fold_{fold}_model.pt")
        torch.save(model.state_dict(), fold_model_path)
        print(f"Saved model weights for fold {fold} to {fold_model_path}")
        fold_results.append({"fold": fold, "test_cindex": test_metrics["c_index"]})

    # Save fold results
    pd.DataFrame(fold_results).to_csv(os.path.join(OUT_DIR, "15fold_results.csv"), index=False)
    print("\nSaved 15-fold CV results to", OUT_DIR)

if __name__ == "__main__":
    main()