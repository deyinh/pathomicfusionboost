import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = "./plots"
N_FOLDS = 15
N_EPOCHS = 30

all_train_losses = []
all_test_losses = []
all_train_cidx = []
all_test_cidx = []

for fold in range(1, N_FOLDS + 1):
    csv_path = os.path.join("./model_checkpoints/clinical_branch", f"fold_{fold}_metrics.csv")
    df = pd.read_csv(csv_path)

    all_train_losses.append(df['train_loss'].values)
    all_test_losses.append(df['test_loss'].values)
    all_train_cidx.append(df['train_cindex'].values)
    all_test_cidx.append(df['test_cindex'].values)

train_losses = np.vstack(all_train_losses)
test_losses = np.vstack(all_test_losses)
train_cidx = np.vstack(all_train_cidx)
test_cidx = np.vstack(all_test_cidx)

mean_train_loss = np.nanmean(train_losses, axis=0)
mean_test_loss = np.nanmean(test_losses, axis=0)
mean_train_cidx = np.nanmean(train_cidx, axis=0)
mean_test_cidx = np.nanmean(test_cidx, axis=0)

epochs = np.arange(1, N_EPOCHS + 1)

# Plot Loss
plt.figure(figsize=(8, 6))
plt.plot(epochs, mean_train_loss, marker='o', label='Train Loss')
plt.plot(epochs, mean_test_loss, marker='o', label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Mean Loss Across 15 Folds')
plt.grid(True)
plt.legend()
plt.savefig(os.path.join(OUT_DIR, "mean_loss_across_folds_clinical.png"), dpi=300)
plt.show()

# Plot C-index
plt.figure(figsize=(8, 6))
plt.plot(epochs, mean_train_cidx, marker='o', label='Train C-index')
plt.plot(epochs, mean_test_cidx, marker='o', label='Test C-index')
plt.xlabel('Epoch')
plt.ylabel('C-index')
plt.title('Mean C-index Across 15 Folds')
plt.grid(True)
plt.legend()
plt.savefig(os.path.join(OUT_DIR, "mean_cindex_across_folds_clinical.png"), dpi=300)
plt.show()