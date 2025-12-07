import numpy as np
import pickle
import torch
import torch.nn as nn

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

class ClinicalNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, output_dim: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),  # Wider layer
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(hidden_dim, output_dim),
            nn.ReLU()
        )
        print(f"ClinicalNet: {input_dim} → {hidden_dim} → {output_dim}")

    def forward(self, x):
        return self.network(x)

def test_clinical_network_basic():
    print("="*60)
    print("BASIC CLINICAL NETWORK TEST")
    print("="*60)

    with open("data/clinical_map.pkl", "rb") as f:
        clinical_data = pickle.load(f)

    with open("data/survival_labels.pkl", "rb") as f:
        survival_labels = pickle.load(f)

    clinical_dim = len(clinical_data["feature_names"])
    n_patients = len(clinical_data["patient_ids"])

    clinical_net = ClinicalNet(clinical_dim, 128, 64)

    # Test forward pass
    patient_ids = list(clinical_data["mapping"].keys())[:5]
    sample_features = np.array([clinical_data["mapping"][pid] for pid in patient_ids])

    x = torch.FloatTensor(sample_features)

    with torch.no_grad():
        y = clinical_net(x)

    if y.std() > 0.01:
        print("Good variance")
    else:
        print("Low variance - network might not be learning")

    X = []
    y_binary = []  # 0 = alive, 1 = dead

    for pid in clinical_data["patient_ids"]:
        if pid in clinical_data["mapping"] and pid in survival_labels:
            X.append(clinical_data["mapping"][pid])
            _, event = survival_labels[pid]
            y_binary.append(event)

    X = np.array(X)
    y_binary = np.array(y_binary)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_binary, test_size=0.2, random_state=42
    )

    X_train_t = torch.FloatTensor(X_train)
    X_test_t = torch.FloatTensor(X_test)

    with torch.no_grad():
        X_train_proc = clinical_net(X_train_t).numpy()
        X_test_proc = clinical_net(X_test_t).numpy()

    lr_proc = LogisticRegression(max_iter=1000, random_state=42)
    lr_proc.fit(X_train_proc, y_train)
    y_pred_proc = lr_proc.predict(X_test_proc)
    acc_proc = accuracy_score(y_test, y_pred_proc)

    print(f"Processed features accuracy: {acc_proc:.3f}")
    return clinical_net, acc_proc

clinical_net, acc_proc = test_clinical_network_basic()