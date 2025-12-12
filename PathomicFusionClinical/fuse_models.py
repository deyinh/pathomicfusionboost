import os
import pickle
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from lifelines.utils import concordance_index
import matplotlib.pyplot as plt


class FusionModel(nn.Module):
    def __init__(self, pathomic_dim, clin_dim):
        super().__init__()
        self.final_layer = nn.Linear(pathomic_dim + clin_dim, 1)

    def forward(self, h_path, h_clin):
        h_fused = torch.cat([h_path, h_clin], dim=1)
        risk = self.final_layer(h_fused).squeeze(-1)
        return risk


def cox_loss(risk, durations, events):
    risk_exp = torch.exp(risk)
    hazard_ratio = torch.log(torch.cumsum(risk_exp.flip(dims=(0,)), dim=0).flip(dims=(0,)))
    loss = -torch.sum((risk - hazard_ratio) * events)
    return loss / events.sum()

h_path_df = pd.read_csv("data/TCGA_GBMLGG/emb_img_surv.csv")
h_path_numeric_cols = h_path_df.select_dtypes(include=[np.number]).columns.tolist()
h_path_df_numeric = h_path_df[h_path_numeric_cols]
h_path_df['tcga_id_simple'] = h_path_df['TCGA ID full'].apply(lambda x: '-'.join(x.split('-')[:3]))
sample_ids = h_path_df['tcga_id_simple'].tolist()
h_path_np = h_path_df_numeric.values.astype(np.float32)

with open("data/TCGA_GBMLGG/clinical_map.pkl", "rb") as f:
    clinical_map_data = pickle.load(f)
clinical_mapping = clinical_map_data['mapping']

clin_features, clin_ids, idx_mask = [], [], []
for i, sid in enumerate(sample_ids):
    if sid in clinical_mapping:
        clin_features.append(clinical_mapping[sid])
        clin_ids.append(sid)
        idx_mask.append(i)

h_path = torch.tensor(h_path_np[idx_mask], dtype=torch.float32)
h_clin = torch.tensor(np.stack(clin_features, axis=0).astype(np.float32), dtype=torch.float32)

with open("data/TCGA_GBMLGG/survival_labels.pkl", "rb") as f:
    survival_labels = pickle.load(f)

durations, events = [], []
for sid in clin_ids:
    dur, ev = survival_labels[sid]
    durations.append(dur)
    events.append(ev)

durations = torch.tensor(np.array(durations, dtype=np.float32))
events = torch.tensor(np.array(events, dtype=np.float32))

splits_df = pd.read_csv("data/TCGA_GBMLGG/pnas_splits.csv", index_col=0)  # index = TCGA ID
num_folds = 15

all_fold_metrics = []

for fold in range(1, num_folds + 1):
    print(f"\n=== Fold {fold} ===")

    train_ids = splits_df[splits_df[f'Randomization - {fold}'] == 'Train'].index.tolist()
    test_ids = splits_df[splits_df[f'Randomization - {fold}'] == 'Test'].index.tolist()

    train_mask = [sid in train_ids for sid in clin_ids]
    test_mask = [sid in test_ids for sid in clin_ids]

    h_path_train = h_path[train_mask]
    h_clin_train = h_clin[train_mask]
    durations_train = durations[train_mask]
    events_train = events[train_mask]

    h_path_test = h_path[test_mask]
    h_clin_test = h_clin[test_mask]
    durations_test = durations[test_mask]
    events_test = events[test_mask]

    pathomic_dim = h_path.shape[1]
    clin_dim = h_clin.shape[1]
    fusion_model = FusionModel(pathomic_dim, clin_dim)
    optimizer = torch.optim.Adam(fusion_model.parameters(), lr=1e-3)
    num_epochs = 100

    train_losses, test_losses, cidx_train, cidx_test = [], [], [], []

    for epoch in range(num_epochs):
        fusion_model.train()
        optimizer.zero_grad()
        risk_train = fusion_model(h_path_train, h_clin_train)
        loss = cox_loss(risk_train, durations_train, events_train)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        fusion_model.eval()
        with torch.no_grad():
            risk_train_eval = fusion_model(h_path_train, h_clin_train)
            risk_test_eval = fusion_model(h_path_test, h_clin_test)

            loss_test_eval = cox_loss(risk_test_eval, durations_test, events_test)
            test_losses.append(loss_test_eval.item())

            c_train = concordance_index(durations_train.numpy(), -risk_train_eval.numpy(), events_train.numpy())
            c_test = concordance_index(durations_test.numpy(), -risk_test_eval.numpy(), events_test.numpy())

            cidx_train.append(c_train)
            cidx_test.append(c_test)

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1}/{num_epochs} - Loss: {loss.item():.4f}, Train C-index: {c_train:.4f}, Test C-index: {c_test:.4f}")

    os.makedirs("model_checkpoints/fusion_cv", exist_ok=True)
    torch.save(fusion_model.state_dict(), f"model_checkpoints/fusion_cv/fusion_model_fold{fold}.pt")

    all_fold_metrics.append({
        'fold': fold,
        'train_losses': train_losses,
        'test_losses': test_losses,
        'cidx_train': cidx_train,
        'cidx_test': cidx_test
    })

num_epochs = len(all_fold_metrics[0]['train_losses'])
train_loss_mean = np.mean([m['train_losses'] for m in all_fold_metrics], axis=0)
test_loss_mean  = np.mean([m['test_losses'] for m in all_fold_metrics], axis=0)
train_c_mean    = np.mean([m['cidx_train'] for m in all_fold_metrics], axis=0)
test_c_mean     = np.mean([m['cidx_test'] for m in all_fold_metrics], axis=0)
epochs = np.arange(1, num_epochs + 1)

plt.figure(figsize=(8, 5))
plt.plot(epochs, train_loss_mean, label='Train Loss', color='blue')
plt.plot(epochs, test_loss_mean, label='Test Loss', color='orange')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Mean Loss Across 15 Folds')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("plots/fusion_mean_loss.png")
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(epochs, train_c_mean, label='Train C-Index', color='blue')
plt.plot(epochs, test_c_mean, label='Test C-Index', color='orange')
plt.xlabel('Epoch')
plt.ylabel('C-Index')
plt.title('Mean C-Index Across 15 Folds')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("plots/fusion_mean_cindex.png")
plt.show()
