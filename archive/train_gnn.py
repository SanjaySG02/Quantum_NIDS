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
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

def load_and_preprocess_data(csv_path, sample_size):
    print("Loading merged dataset...")
    df = pd.read_csv(csv_path)

    # Drop index/unnamed columns if they exist
    if df.columns[0] == '' or df.columns[0].startswith('Unnamed'):
        df = df.drop(df.columns[0], axis=1)

    print(f"Original shape: {df.shape}")

    # Downsample if specified
    if sample_size > 0:
        half_sample = sample_size // 2
        df_normal = df[df['Label'] == 0]
        df_attack = df[df['Label'] == 1]
        df_normal = df_normal.sample(n=min(half_sample, len(df_normal)), random_state=42)
        df_attack = df_attack.sample(n=min(half_sample, len(df_attack)), random_state=42)
        df = pd.concat([df_normal, df_attack], axis=0, ignore_index=True)
        print(f"Sampled shape: {df.shape}")

    # Target variable is "Label"
    y = df['Label'].values
    X = df.drop('Label', axis=1)

    # Clean numeric columns (handle NaNs and Inf)
    for col in X.columns:
        if X[col].dtype == object:
            # If there are categorical columns, map them
            X[col] = pd.factorize(X[col])[0]
        else:
            # Replace Inf with NaN, then fill NaN with median
            X[col] = X[col].replace([np.inf, -np.inf], np.nan)
            col_median = X[col].median()
            if pd.isna(col_median):
                col_median = 0
            X[col] = X[col].fillna(col_median)

    # Scale numeric features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y

def build_knn_graph(X, k):
    print(f"Constructing k-NN similarity graph (k={k})...")
    # Build k-NN adjacency matrix (binary, directed or undirected)
    # Using 'connectivity' mode gives a binary matrix of edges
    adj = kneighbors_graph(X, n_neighbors=k, mode='connectivity', include_self=True)
    
    # Make the adjacency matrix symmetric (undirected graph representation)
    adj = adj + adj.T
    adj.data = np.ones_like(adj.data)  # Keep it binary
    
    # Compute normalized Laplacian D^-0.5 * A * D^-0.5
    row_sum = np.array(adj.sum(axis=1)).flatten()
    # Avoid division by zero
    row_sum_inv_sqrt = np.power(row_sum, -0.5, where=row_sum > 0, out=np.zeros_like(row_sum))
    
    # Multiply D^-0.5 * A * D^-0.5
    d_inv_sqrt = scipy_sparse_diag(row_sum_inv_sqrt)
    adj_normalized = d_inv_sqrt.dot(adj).dot(d_inv_sqrt).tocoo()
    
    # Convert to PyTorch sparse tensor
    indices = torch.from_numpy(np.vstack((adj_normalized.row, adj_normalized.col))).long()
    values = torch.from_numpy(adj_normalized.data).float()
    shape = torch.Size(adj_normalized.shape)
    
    adj_tensor = torch.sparse_coo_tensor(indices, values, shape)
    return adj_tensor

def scipy_sparse_diag(val):
    # A simple helper to avoid importing scipy directly if possible, 
    # but scipy is installed with scikit-learn.
    import scipy.sparse as sp
    return sp.diags(val)

class SparseGCNConv(nn.Module):
    """
    Custom Graph Convolutional Layer using PyTorch sparse matrix multiplication.
    This eliminates complex external package dependencies (like torch-geometric).
    """
    def __init__(self, in_features, out_features):
        super(SparseGCNConv, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, adj):
        # x is dense tensor [num_nodes, in_features]
        # adj is sparse tensor [num_nodes, num_nodes]
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

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    default_csv = os.path.join(base_dir, 'CTU13_Merged_Traffic.csv')
    parser = argparse.ArgumentParser(description="Train a classical GNN on CTU-13 dataset.")
    parser.add_argument('--csv_path', type=str, default=default_csv, help='Path to merged traffic csv')
    parser.add_argument('--sample_size', type=int, default=10000, help='Total sample size to downsample to (0 for all)')
    parser.add_argument('--k_neighbors', type=int, default=5, help='Number of neighbors in k-NN graph')
    parser.add_argument('--hidden_dim', type=int, default=32, help='GCN hidden layers dimension')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay L2 regularization')
    args = parser.parse_args()

    # Load and clean data
    X, y = load_and_preprocess_data(args.csv_path, args.sample_size)
    num_nodes = X.shape[0]
    num_features = X.shape[1]
    
    # Build graph
    adj = build_knn_graph(X, args.k_neighbors)

    # Convert node features and labels to PyTorch tensors
    features = torch.FloatTensor(X)
    labels = torch.LongTensor(y.copy())

    # Train/Val/Test Splits (Indices split)
    np.random.seed(42)
    indices = np.random.permutation(num_nodes)
    train_end = int(0.7 * num_nodes)
    val_end = int(0.85 * num_nodes)

    train_idx = torch.LongTensor(indices[:train_end])
    val_idx = torch.LongTensor(indices[train_end:val_end])
    test_idx = torch.LongTensor(indices[val_end:])

    print(f"Split sizes - Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

    # Initialize GCN model
    # Output dim is 2 because this is a binary classification task
    model = GCN(in_features=num_features, hidden_features=args.hidden_dim, out_features=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    # Tracking metrics
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    print("Starting training...")
    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        output = model(features, adj)
        loss_train = criterion(output[train_idx], labels[train_idx])
        loss_train.backward()
        optimizer.step()

        # Calculate train accuracy
        preds_train = output[train_idx].max(1)[1]
        acc_train = preds_train.eq(labels[train_idx]).double().mean().item()

        # Evaluate on validation split
        model.eval()
        with torch.no_grad():
            output = model(features, adj)
            loss_val = criterion(output[val_idx], labels[val_idx])
            preds_val = output[val_idx].max(1)[1]
            acc_val = preds_val.eq(labels[val_idx]).double().mean().item()

        train_losses.append(loss_train.item())
        val_losses.append(loss_val.item())
        train_accs.append(acc_train)
        val_accs.append(acc_val)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} | Train Loss: {loss_train.item():.4f} | Train Acc: {acc_train:.4f} | Val Loss: {loss_val.item():.4f} | Val Acc: {acc_val:.4f}")

    training_time = time.time() - start_time
    print("Training finished. Evaluating on test set...")
    model.eval()
    with torch.no_grad():
        output = model(features, adj)
        preds_test = output[test_idx].max(1)[1].cpu().numpy()
        labels_test = labels[test_idx].cpu().numpy()

    # Print performance report
    report = classification_report(labels_test, preds_test, target_names=['Normal', 'Attack'])
    print("\nTest Classification Report:")
    print(report)

    conf_mat = confusion_matrix(labels_test, preds_test)
    print("Confusion Matrix:")
    print(conf_mat)
    print(f"\nTraining Time: {training_time:.4f} seconds")

    # Plot metrics
    plt.figure(figsize=(12, 5))
    
    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Loss History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Val Accuracy')
    plt.title('Accuracy History')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('training_history.png')
    print("Saved training history plot to training_history.png")

    # Confusion matrix plot
    plt.figure(figsize=(6, 5))
    try:
        import seaborn as sns
        sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Attack'], yticklabels=['Normal', 'Attack'])
    except ImportError:
        # Fallback plot using matplotlib directly if seaborn is not installed
        plt.imshow(conf_mat, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Test Confusion Matrix')
        plt.colorbar()
        tick_marks = np.arange(2)
        plt.xticks(tick_marks, ['Normal', 'Attack'])
        plt.yticks(tick_marks, ['Normal', 'Attack'])
        
        # Labeling the matrix cells
        thresh = conf_mat.max() / 2.
        for i in range(2):
            for j in range(2):
                plt.text(j, i, format(conf_mat[i, j], 'd'),
                         horizontalalignment="center",
                         color="white" if conf_mat[i, j] > thresh else "black")
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    print("Saved confusion matrix plot to confusion_matrix.png")

if __name__ == '__main__':
    main()
