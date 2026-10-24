import os
import time
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, matthews_corrcoef, roc_curve
)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ClassicalNIDS")

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

def preprocess_and_split_data(csv_path: str, k_features: int = 4):
    """
    Loads, cleans, encodes, splits, and scales the CTU-13 dataset.
    Splits:
      - Train: 70% Malicious, 30% Benign
      - Val:   50% Malicious, 50% Benign
      - Test:  10% Malicious, 90% Benign
    """
    logger.info("Loading merged dataset from %s...", csv_path)
    df = pd.read_csv(csv_path)

    # Drop index/unnamed columns if they exist
    if df.columns[0] == '' or df.columns[0].startswith('Unnamed'):
        df = df.drop(df.columns[0], axis=1)

    # Re-map/rename Label column to label (Normal = 0, Attack = 1)
    if 'Label' in df.columns:
        df['label'] = df['Label']
        df = df.drop('Label', axis=1)
    elif 'label' not in df.columns:
        raise ValueError("Target variable 'Label' not found in dataset columns.")

    # 1. Missing Value Handling
    df = df.dropna(subset=['label'])
    
    # 2. Duplicate Removal
    initial_shape = df.shape
    df = df.drop_duplicates()
    logger.info("Dropped duplicates. Initial shape: %s, Cleaned shape: %s", initial_shape, df.shape)

    # Remove original Label column if present
    if 'Label' in df.columns:
        df = df.drop('Label', axis=1)

    # Encode categorical features and build vocabularies
    vocabularies = {}
    for col in df.columns:
        if col == 'label':
            continue
        if df[col].dtype == object:
            df[col] = df[col].astype(str).replace('-', 'MISSING')
            categories, uniques = pd.factorize(df[col])
            df[col] = categories
            vocabularies[col] = {str(val): int(idx) for idx, val in enumerate(uniques)}
        else:
            # Clean numerical columns
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            col_median = df[col].median()
            if pd.isna(col_median):
                col_median = 0.0
            df[col] = df[col].fillna(col_median)

    # Separate classes
    df_benign = df[df['label'] == 0]
    df_malicious = df[df['label'] == 1]
    
    n_benign = len(df_benign)
    n_malicious = len(df_malicious)
    
    logger.info("Class Counts - Benign: %d, Malicious: %d", n_benign, n_malicious)

    # Split malicious indices (60% Train, 20% Val, 20% Test)
    mal_indices = np.random.permutation(df_malicious.index)
    train_mal_count = int(0.6 * n_malicious)
    val_mal_count = int(0.2 * n_malicious)
    test_mal_count = n_malicious - train_mal_count - val_mal_count

    train_mal_idx = mal_indices[:train_mal_count]
    val_mal_idx = mal_indices[train_mal_count:train_mal_count + val_mal_count]
    test_mal_idx = mal_indices[train_mal_count + val_mal_count:]

    # Calculate required benign counts to meet target ratios
    # Train: 30% Benign, 70% Malicious -> Benign = Malicious * (0.3 / 0.7)
    train_ben_count = int(train_mal_count * 3 / 7)
    # Val: 50% Benign, 50% Malicious -> Benign = Malicious
    val_ben_count = val_mal_count
    # Test: 90% Benign, 10% Malicious -> Benign = Malicious * 9
    test_ben_count = test_mal_count * 9

    total_benign_needed = train_ben_count + val_ben_count + test_ben_count
    logger.info("Benign samples needed: %d (Available: %d)", total_benign_needed, n_benign)

    if total_benign_needed > n_benign:
        # Scale down if benign count is insufficient
        scale_factor = n_benign / total_benign_needed
        train_ben_count = int(train_ben_count * scale_factor)
        val_ben_count = int(val_ben_count * scale_factor)
        test_ben_count = int(test_ben_count * scale_factor)
        
        train_mal_idx = train_mal_idx[:int(train_ben_count * 7 / 3)]
        val_mal_idx = val_mal_idx[:val_ben_count]
        test_mal_idx = test_mal_idx[:int(test_ben_count / 9)]

    # Draw benign indices
    ben_indices = np.random.permutation(df_benign.index)
    train_ben_idx = ben_indices[:train_ben_count]
    val_ben_idx = ben_indices[train_ben_count:train_ben_count + val_ben_count]
    test_ben_idx = ben_indices[train_ben_count + val_ben_count:train_ben_count + val_ben_count + test_ben_count]

    # Combine partitions
    train_df = pd.concat([df.loc[train_mal_idx], df.loc[train_ben_idx]]).sample(frac=1, random_state=RANDOM_SEED)
    val_df = pd.concat([df.loc[val_mal_idx], df.loc[val_ben_idx]]).sample(frac=1, random_state=RANDOM_SEED)
    test_df = pd.concat([df.loc[test_mal_idx], df.loc[test_ben_idx]]).sample(frac=1, random_state=RANDOM_SEED)

    # Separate features and labels
    X_train = train_df.drop('label', axis=1)
    y_train = train_df['label'].values
    X_val = val_df.drop('label', axis=1)
    y_val = val_df['label'].values
    X_test = test_df.drop('label', axis=1)
    y_test = test_df['label'].values

    # Determine feature importance via Random Forest on full train set to select top K
    logger.info("Evaluating feature importances with Random Forest...")
    rf_feat = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1)
    rf_feat.fit(X_train, y_train)
    
    importances = rf_feat.feature_importances_
    indices = np.argsort(importances)[::-1]
    top_k_features = X_train.columns[indices[:k_features]].tolist()
    logger.info("Automatically selected Top-%d features: %s", k_features, top_k_features)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data") if os.path.isdir(os.path.join(base_dir, "data")) else base_dir
    artifacts_dir = os.path.join(base_dir, "artifacts") if os.path.isdir(os.path.join(base_dir, "artifacts")) else base_dir
    plots_dir = os.path.join(base_dir, "plots") if os.path.isdir(os.path.join(base_dir, "plots")) else base_dir

    # Plot and save feature importances
    plt.figure(figsize=(10, 6))
    plt.title("Random Forest Feature Importances")
    plt.bar(range(k_features), importances[indices[:k_features]], align="center", color="#4361ee")
    plt.xticks(range(k_features), top_k_features, rotation=15)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "feature_importance.png"))
    plt.close()

    # Filter features
    X_train = X_train[top_k_features]
    X_val = X_val[top_k_features]
    X_test = X_test[top_k_features]

    # Scale/Normalize selected features to [0, pi] for QML compatibility
    logger.info("Scaling features to [0, pi] range...")
    scaler = MinMaxScaler(feature_range=(0.0, np.pi))
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Save preprocessing metadata for real-time live sniffing
    meta = {
        "selected_features": top_k_features,
        "vocabularies": {col: vocabularies[col] for col in top_k_features if col in vocabularies},
        "scaler_min": scaler.min_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "scaler_data_min": scaler.data_min_.tolist(),
        "scaler_data_max": scaler.data_max_.tolist()
    }
    with open(os.path.join(artifacts_dir, "preprocessing_meta.json"), "w") as f:
        json.dump(meta, f)

    # Save selected features to file
    with open(os.path.join(artifacts_dir, "selected_features.json"), "w") as f:
        json.dump(top_k_features, f)

    # Reconstruct as DataFrames to save csv files
    train_processed = pd.DataFrame(X_train_scaled, columns=top_k_features)
    train_processed['label'] = y_train
    train_processed.to_csv(os.path.join(data_dir, "train_processed.csv"), index=False)

    val_processed = pd.DataFrame(X_val_scaled, columns=top_k_features)
    val_processed['label'] = y_val
    val_processed.to_csv(os.path.join(data_dir, "val_processed.csv"), index=False)

    test_processed = pd.DataFrame(X_test_scaled, columns=top_k_features)
    test_processed['label'] = y_test
    test_processed.to_csv(os.path.join(data_dir, "test_processed.csv"), index=False)

    logger.info("Processed datasets saved to %s. Train shape: %s, Val shape: %s, Test shape: %s", 
                data_dir, train_processed.shape, val_processed.shape, test_processed.shape)

    return X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test, top_k_features

def evaluate_classifier(model, X_train, y_train, X_test, y_test, name: str):
    """
    Fits and evaluates a model, returning comprehensive evaluation metrics.
    """
    logger.info("Training %s...", name)
    start_train = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_train

    start_inf = time.time()
    preds = model.predict(X_test)
    inf_time = (time.time() - start_inf) / len(X_test)

    # Compute probabilities for ROC AUC
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_test)[:, 1]
    else:
        probs = model.decision_function(X_test)

    # Metrics
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    roc_auc = roc_auc_score(y_test, probs)
    mcc = matthews_corrcoef(y_test, preds)

    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    metrics = {
        "Model": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1,
        "ROC-AUC": roc_auc,
        "FPR": fpr,
        "FNR": fnr,
        "MCC": mcc,
        "Training Time (s)": train_time,
        "Inference Time (s/sample)": inf_time
    }
    
    return metrics, preds, probs

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    artifacts_dir = os.path.join(base_dir, "artifacts") if os.path.isdir(os.path.join(base_dir, "artifacts")) else base_dir
    plots_dir = os.path.join(base_dir, "plots") if os.path.isdir(os.path.join(base_dir, "plots")) else base_dir

    csv_candidates = [
        os.path.join(base_dir, "data", "CTU13_Merged_Traffic.csv"),
        os.path.join(base_dir, "CTU13_Merged_Traffic.csv"),
        "data/CTU13_Merged_Traffic.csv",
        "CTU13_Merged_Traffic.csv"
    ]
    csv_path = next((p for p in csv_candidates if os.path.exists(p)), None)
    if not csv_path:
        logger.error("CTU-13 merged dataset not found. Please ensure CTU13_Merged_Traffic.csv is present in data/ or root.")
        return

    # Process and split
    X_train, y_train, X_val, y_val, X_test, y_test, features = preprocess_and_split_data(csv_path)

    # Instantiate models
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_SEED),
        "HistGradientBoosting": HistGradientBoostingClassifier(random_state=RANDOM_SEED),
        "MLP": MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=200, random_state=RANDOM_SEED)
    }

    all_metrics = []
    predictions = {}
    probabilities = {}

    for name, model in models.items():
        metrics, preds, probs = evaluate_classifier(model, X_train, y_train, X_test, y_test, name)
        all_metrics.append(metrics)
        predictions[name] = preds
        probabilities[name] = probs

    # Output metrics to CSV
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(os.path.join(artifacts_dir, "classical_metrics.csv"), index=False)
    logger.info("Classical metrics written to %s:", os.path.join(artifacts_dir, "classical_metrics.csv"))
    print(metrics_df.to_string(index=False))

    # --- PLOTTING ---
    logger.info("Generating evaluation plots...")
    
    # 1. Confusion Matrix Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    for idx, (name, preds) in enumerate(predictions.items()):
        cm = confusion_matrix(y_test, preds)
        ax = axes[idx]
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.set_title(f"{name} Confusion Matrix")
        fig.colorbar(im, ax=ax)
        classes = ['Benign', 'Malicious']
        tick_marks = np.arange(len(classes))
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(classes)
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(classes)
        
        # Label cells
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        ax.set_ylabel('Actual Label')
        ax.set_xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "confusion_matrix_classical.png"))
    plt.close()

    # 2. ROC Curve Plot
    plt.figure(figsize=(10, 8))
    for name, probs in probabilities.items():
        fpr_pts, tpr_pts, _ = roc_curve(y_test, probs)
        plt.plot(fpr_pts, tpr_pts, label=f"{name} (AUC = {roc_auc_score(y_test, probs):.4f})")
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves - Classical Models')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(plots_dir, "roc_curve_classical.png"))
    plt.close()

    # 3. Model Comparison Plot (Accuracy vs F1-Score)
    plt.figure(figsize=(10, 6))
    x_indices = np.arange(len(metrics_df))
    bar_width = 0.35
    plt.bar(x_indices - bar_width/2, metrics_df['Accuracy'] * 100, bar_width, label='Accuracy (%)', color='#3f37c9')
    plt.bar(x_indices + bar_width/2, metrics_df['F1-Score'] * 100, bar_width, label='F1-Score (%)', color='#4cc9f0')
    plt.title('Classical Models Comparison')
    plt.xticks(x_indices, metrics_df['Model'])
    plt.ylabel('Percentage')
    plt.ylim(80, 105)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "model_comparison_classical.png"))
    plt.close()

    logger.info("Classical NIDS script completed successfully.")

if __name__ == '__main__':
    main()
