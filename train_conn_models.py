import os
import argparse
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import scipy.sparse as sp

# =====================================================================
# 1. DATA LOADING AND PREPROCESSING
# =====================================================================

def load_and_preprocess_conn_data(csv_path):
    print(f"Loading Zeek logs from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Dataset shape: {df.shape}")

    # Drop non-informative features to prevent overfitting on IPs and IDs
    drop_cols = ['uid', 'id.orig_h', 'id.resp_h', 'detailed-label']
    df = df.drop([col for col in drop_cols if col in df.columns], axis=1)

    # Encode label: 'benign' -> 0, 'malicious' -> 1
    if 'label' in df.columns:
        df['label'] = df['label'].map({'benign': 0, 'malicious': 1})
    
    y = df['label'].values
    X = df.drop('label', axis=1)

    # Encode categorical columns
    categorical_cols = ['proto', 'service', 'conn_state', 'history', 'local_orig', 'local_resp', 'tunnel_parents']
    for col in X.columns:
        if col in categorical_cols or X[col].dtype == object:
            # Replace missing markers '-' with a standard category 'MISSING'
            X[col] = X[col].astype(str).replace('-', 'MISSING')
            X[col] = pd.factorize(X[col])[0]
        else:
            # Clean numerical columns
            X[col] = X[col].replace([np.inf, -np.inf], np.nan)
            col_median = X[col].median()
            if pd.isna(col_median):
                col_median = 0
            X[col] = X[col].fillna(col_median)

    # Scale all features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y

# =====================================================================
# 2. GNN (SPARSE GCN LAYER)
# =====================================================================

def build_knn_graph(X, k):
    print(f"Constructing k-NN similarity graph (k={k})...")
    adj = kneighbors_graph(X, n_neighbors=k, mode='connectivity', include_self=True)
    adj = adj + adj.T
    adj.data = np.ones_like(adj.data)
    
    row_sum = np.array(adj.sum(axis=1)).flatten()
    row_sum_inv_sqrt = np.power(row_sum, -0.5, where=row_sum > 0, out=np.zeros_like(row_sum))
    d_inv_sqrt = sp.diags(row_sum_inv_sqrt)
    adj_normalized = d_inv_sqrt.dot(adj).dot(d_inv_sqrt).tocoo()
    
    indices = torch.from_numpy(np.vstack((adj_normalized.row, adj_normalized.col))).long()
    values = torch.from_numpy(adj_normalized.data).float()
    shape = torch.Size(adj_normalized.shape)
    
    adj_tensor = torch.sparse_coo_tensor(indices, values, shape)
    return adj_tensor

class SparseGCNConv(nn.Module):
    def __init__(self, in_features, out_features):
        super(SparseGCNConv, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, adj):
        support = torch.mm(x, self.weight)
        output = torch.sparse.mm(adj, support)
        return output

class GCN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, dropout=0.5):
        super(GCN, self).__init__()
        self.gcn1 = SparseGCNConv(in_features, hidden_features)
        self.gcn2 = SparseGCNConv(hidden_features, hidden_features)
        self.fc = nn.Linear(hidden_features, out_features)
        self.dropout = dropout

    def forward(self, x, adj):
        x = self.gcn1(x, adj)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gcn2(x, adj)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc(x)
        return x

# =====================================================================
# 3. LSTM SEQUENCE FORMULATION
# =====================================================================

def create_sequences(X, y, seq_length=5):
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_length + 1):
        X_seq.append(X[i:i+seq_length])
        y_seq.append(y[i+seq_length-1])
    return np.array(X_seq), np.array(y_seq)

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes=2, num_layers=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# =====================================================================
# 4. TRAINING AND EVALUATION PIPELINE
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Train intrusion detection models on balanced new_conn_file.csv.")
    parser.add_argument('--csv_path', type=str, default=r'd:\Mini Project\new_conn_file.csv')
    parser.add_argument('--epochs', type=int, default=50, help='Epochs to train GNN and LSTM (default: 50)')
    parser.add_argument('--k_neighbors', type=int, default=5, help='k-NN graph neighbors (default: 5)')
    args = parser.parse_args()

    # Preprocess
    X, y = load_and_preprocess_conn_data(args.csv_path)
    num_nodes = X.shape[0]
    num_features = X.shape[1]

    # Split (70% Train, 15% Val, 15% Test)
    np.random.seed(42)
    indices = np.random.permutation(num_nodes)
    train_end = int(0.7 * num_nodes)
    val_end = int(0.85 * num_nodes)

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    results = {}

    # -----------------------------------------------------------------
    # Model 1: Random Forest
    # -----------------------------------------------------------------
    print("\n--- Training Random Forest ---")
    start_time = time.time()
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_time = time.time() - start_time
    
    rf_preds = rf.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_preds)
    p, r, f, _ = precision_recall_fscore_support(y_test, rf_preds, average='binary')
    results['Random Forest'] = {'Accuracy': rf_acc, 'Precision': p, 'Recall': r, 'F1-Score': f, 'Time (s)': rf_time}
    print(f"Random Forest finished. Acc: {rf_acc:.4f} | F1: {f:.4f}")

    # -----------------------------------------------------------------
    # Model 2: XGBoost
    # -----------------------------------------------------------------
    print("\n--- Training XGBoost ---")
    start_time = time.time()
    xgb = XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss', n_jobs=-1)
    xgb.fit(X_train, y_train)
    xgb_time = time.time() - start_time
    
    xgb_preds = xgb.predict(X_test)
    xgb_acc = accuracy_score(y_test, xgb_preds)
    p, r, f, _ = precision_recall_fscore_support(y_test, xgb_preds, average='binary')
    results['XGBoost'] = {'Accuracy': xgb_acc, 'Precision': p, 'Recall': r, 'F1-Score': f, 'Time (s)': xgb_time}
    print(f"XGBoost finished. Acc: {xgb_acc:.4f} | F1: {f:.4f}")

    # -----------------------------------------------------------------
    # Model 3: GNN (GCN)
    # -----------------------------------------------------------------
    print("\n--- Training GNN (GCN) ---")
    adj = build_knn_graph(X, args.k_neighbors)
    features_t = torch.FloatTensor(X)
    labels_t = torch.LongTensor(y.copy())
    
    train_idx_t = torch.LongTensor(train_idx)
    test_idx_t = torch.LongTensor(test_idx)

    gnn = GCN(in_features=num_features, hidden_features=32, out_features=2)
    optimizer = torch.optim.Adam(gnn.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()

    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        gnn.train()
        optimizer.zero_grad()
        output = gnn(features_t, adj)
        loss = criterion(output[train_idx_t], labels_t[train_idx_t])
        loss.backward()
        optimizer.step()
    gnn_time = time.time() - start_time

    gnn.eval()
    with torch.no_grad():
        output = gnn(features_t, adj)
        gnn_preds = output[test_idx_t].max(1)[1].numpy()
        
    gnn_acc = accuracy_score(y_test, gnn_preds)
    p, r, f, _ = precision_recall_fscore_support(y_test, gnn_preds, average='binary')
    results['GNN (GCN)'] = {'Accuracy': gnn_acc, 'Precision': p, 'Recall': r, 'F1-Score': f, 'Time (s)': gnn_time}
    print(f"GNN finished. Acc: {gnn_acc:.4f} | F1: {f:.4f}")

    # -----------------------------------------------------------------
    # Model 4: LSTM
    # -----------------------------------------------------------------
    print("\n--- Training LSTM Sequence Model ---")
    seq_length = 5
    X_train_seq, y_train_seq = create_sequences(X_train, y_train, seq_length)
    X_test_seq, y_test_seq = create_sequences(X_test, y_test, seq_length)

    train_x_t = torch.FloatTensor(X_train_seq)
    train_y_t = torch.LongTensor(y_train_seq.copy())
    test_x_t = torch.FloatTensor(X_test_seq)

    lstm = LSTMModel(input_dim=num_features, hidden_dim=32, num_classes=2)
    optimizer = torch.optim.Adam(lstm.parameters(), lr=0.005)
    criterion = nn.CrossEntropyLoss()

    train_dataset = torch.utils.data.TensorDataset(train_x_t, train_y_t)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        lstm.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            output = lstm(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
    lstm_time = time.time() - start_time

    lstm.eval()
    with torch.no_grad():
        output = lstm(test_x_t)
        lstm_preds = output.max(1)[1].numpy()
        
    lstm_acc = accuracy_score(y_test_seq, lstm_preds)
    p, r, f, _ = precision_recall_fscore_support(y_test_seq, lstm_preds, average='binary')
    results['LSTM'] = {'Accuracy': lstm_acc, 'Precision': p, 'Recall': r, 'F1-Score': f, 'Time (s)': lstm_time}
    print(f"LSTM finished. Acc: {lstm_acc:.4f} | F1: {f:.4f}")

    # =====================================================================
    # 5. COMPARISON TABLE & VISUALIZATION
    # =====================================================================
    print("\n" + "="*60)
    print("ZEEK LOG DATASET (new_conn_file.csv) MODEL COMPARISON")
    print("="*60)
    
    df_results = pd.DataFrame(results).T
    print(df_results.to_markdown())

    # Save comparison plot
    plt.figure(figsize=(10, 6))
    models = list(results.keys())
    accuracies = [results[m]['Accuracy'] * 100 for m in models]
    f1_scores = [results[m]['F1-Score'] * 100 for m in models]
    
    x = np.arange(len(models))
    width = 0.35
    
    plt.bar(x - width/2, accuracies, width, label='Accuracy (%)', color='#1f77b4')
    plt.bar(x + width/2, f1_scores, width, label='F1-Score (%)', color='#2ca02c')
    
    plt.ylabel('Percentage (%)')
    plt.title('Zeek Log Intrusion Detection Model Comparison')
    plt.xticks(x, models)
    plt.ylim(80, 105)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    for i in range(len(models)):
        plt.text(i - width/2, accuracies[i] + 0.5, f"{accuracies[i]:.1f}%", ha='center', va='bottom', fontsize=9)
        plt.text(i + width/2, f1_scores[i] + 0.5, f"{f1_scores[i]:.1f}%", ha='center', va='bottom', fontsize=9)
        
    plt.tight_layout()
    plt.savefig('conn_model_comparison.png')
    print("\nSaved comparison chart to conn_model_comparison.png")

if __name__ == '__main__':
    main()
