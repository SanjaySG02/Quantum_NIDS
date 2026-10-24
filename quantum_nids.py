import os
import time
import json
import logging
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Qiskit imports
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector, state_fidelity
try:
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, depolarizing_error
    AER_AVAILABLE = True
except ImportError:
    AerSimulator = None
    NoiseModel = None
    depolarizing_error = None
    AER_AVAILABLE = False
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, matthews_corrcoef, roc_curve
)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("QuantumNIDS")

# Suppress verbose external package logs
logging.getLogger('qiskit').setLevel(logging.WARNING)
if AER_AVAILABLE:
    logging.getLogger('qiskit_aer').setLevel(logging.WARNING)
logging.getLogger('stevedore').setLevel(logging.WARNING)

# Random seed
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

def get_noisy_simulator():
    """
    Creates an AerSimulator with custom depolarizing error models for noisy simulation.
    """
    if not AER_AVAILABLE:
        return None
    noise_model = NoiseModel()
    # 1% gate error on single qubit gates, 5% gate error on CNOT gates
    error_1 = depolarizing_error(0.01, 1)
    error_2 = depolarizing_error(0.05, 2)
    noise_model.add_all_qubit_quantum_error(error_1, ['ry', 'h'])
    noise_model.add_all_qubit_quantum_error(error_2, ['cx'])
    return AerSimulator(noise_model=noise_model)

def construct_fidelity_circuit(x: np.ndarray, p: np.ndarray) -> QuantumCircuit:
    """
    Builds the Compute-Uncompute fidelity circuit between sample state x and prototype p.
    """
    num_qubits = len(x)
    qc = QuantumCircuit(num_qubits)
    
    # 1. Encode sample x: U(x)
    for i in range(num_qubits):
        qc.ry(x[i], i)
    for i in range(num_qubits):
        qc.h(i)
    for i in range(num_qubits - 1):
        qc.cx(i, i+1)
        
    # 2. Uncompute prototype p: U(p)^+
    for i in range(num_qubits - 2, -1, -1):
        qc.cx(i, i+1)
    for i in range(num_qubits):
        qc.h(i)
    for i in range(num_qubits):
        qc.ry(-p[i], i)
        
    qc.measure_all()
    return qc

def calculate_statevector_fidelity(x: np.ndarray, p: np.ndarray) -> float:
    """
    Computes analytical statevector fidelity (ideal mathematical reference).
    """
    num_qubits = len(x)
    # Circuit for x
    qc_x = QuantumCircuit(num_qubits)
    for i in range(num_qubits):
        qc_x.ry(x[i], i)
    for i in range(num_qubits):
        qc_x.h(i)
    for i in range(num_qubits - 1):
        qc_x.cx(i, i+1)
    sv_x = Statevector.from_instruction(qc_x)

    # Circuit for p
    qc_p = QuantumCircuit(num_qubits)
    for i in range(num_qubits):
        qc_p.ry(p[i], i)
    for i in range(num_qubits):
        qc_p.h(i)
    for i in range(num_qubits - 1):
        qc_p.cx(i, i+1)
    sv_p = Statevector.from_instruction(qc_p)

    return state_fidelity(sv_x, sv_p)

def run_fidelity_estimation(x: np.ndarray, p: np.ndarray, simulator=None, shots: int = 1024) -> float:
    """
    Runs the compute-uncompute circuit on the specified AerSimulator, or falls back to Statevector.
    """
    if simulator is not None and AER_AVAILABLE:
        qc = construct_fidelity_circuit(x, p)
        transpiled_qc = transpile(qc, simulator)
        job = simulator.run(transpiled_qc, shots=shots)
        counts = job.result().get_counts(transpiled_qc)
        zero_state = '0' * len(x)
        return counts.get(zero_state, 0) / shots
    else:
        return calculate_statevector_fidelity(x, p)

def main():
    parser = argparse.ArgumentParser(description="Train and evaluate Hybrid Quantum Similarity Engine NIDS.")
    parser.add_argument('--val_sample_size', type=int, default=1000, help='Validation sample size for threshold tuning (0 for all)')
    parser.add_argument('--test_sample_size', type=int, default=150, help='Test sample size for evaluation (0 for all)')
    parser.add_argument('--shots', type=int, default=1024, help='Number of shots for Aer simulator runs')
    args = parser.parse_args()

    # Helper to resolve file paths across root and subfolders
    def find_file(fname, subfolders=("data", "artifacts", "plots", "")):
        for f in subfolders:
            p = os.path.join(f, fname) if f else fname
            if os.path.exists(p):
                return p
        return fname

    train_path = find_file("train_processed.csv", ("data", ""))
    val_path = find_file("val_processed.csv", ("data", ""))
    test_path = find_file("test_processed.csv", ("data", ""))
    feat_path = find_file("selected_features.json", ("artifacts", ""))

    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_candidates = [os.path.join(base_dir, "data", "train_processed.csv"), os.path.join(base_dir, "train_processed.csv"), "data/train_processed.csv", "train_processed.csv"]
    val_candidates = [os.path.join(base_dir, "data", "val_processed.csv"), os.path.join(base_dir, "val_processed.csv"), "data/val_processed.csv", "val_processed.csv"]
    test_candidates = [os.path.join(base_dir, "data", "test_processed.csv"), os.path.join(base_dir, "test_processed.csv"), "data/test_processed.csv", "test_processed.csv"]
    feat_candidates = [os.path.join(base_dir, "artifacts", "selected_features.json"), os.path.join(base_dir, "selected_features.json"), "artifacts/selected_features.json", "selected_features.json"]

    train_path = next((p for p in train_candidates if os.path.exists(p)), None)
    val_path = next((p for p in val_candidates if os.path.exists(p)), None)
    test_path = next((p for p in test_candidates if os.path.exists(p)), None)
    feat_path = next((p for p in feat_candidates if os.path.exists(p)), None)

    if not (train_path and val_path and test_path and feat_path):
        logger.error("Missing processed dataset files. Please run classical_nids.py first.")
        return

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    with open(feat_path, "r") as f:
        selected_features = json.load(f)

    logger.info("Loaded processed datasets from %s. Selected features: %s", train_path, selected_features)

    artifacts_dir = os.path.join(base_dir, "artifacts") if os.path.isdir(os.path.join(base_dir, "artifacts")) else base_dir
    plots_dir = os.path.join(base_dir, "plots") if os.path.isdir(os.path.join(base_dir, "plots")) else base_dir

    # Class separation on train set to create prototype states
    train_normal = train_df[train_df['label'] == 0][selected_features].values
    train_attack = train_df[train_df['label'] == 1][selected_features].values

    # Mean prototype vector
    normal_prototype = np.mean(train_normal, axis=0)
    attack_prototype = np.mean(train_attack, axis=0)
    logger.info("Normal Prototype Mean Vector: %s", normal_prototype)
    logger.info("Attack Prototype Mean Vector: %s", attack_prototype)

    # Generate and save Quantum Circuit diagram
    sample_ref = train_normal[0]
    qc_ref = construct_fidelity_circuit(sample_ref, normal_prototype)
    logger.info("Saving quantum circuit diagram...")
    try:
        fig = qc_ref.draw(output='mpl')
        fig.savefig(os.path.join(plots_dir, "quantum_circuit.png"))
        plt.close(fig)
    except Exception as e:
        logger.warning("Could not draw circuit using MPL. Outputting as text format. Error: %s", e)
        with open(os.path.join(plots_dir, "quantum_circuit.txt"), "w", encoding="utf-8") as f:
            f.write(str(qc_ref.draw(output='text')))

    # Calculate circuit metadata
    num_qubits = len(selected_features)
    circuit_depth = qc_ref.depth()
    gate_count = qc_ref.count_ops()
    logger.info("Circuit Properties - Qubits: %d, Depth: %d, Ops: %s", num_qubits, circuit_depth, dict(gate_count))

    # --- ADAPTIVE THRESHOLD ON VALIDATION SET ---
    logger.info("Calculating optimal similarity threshold on validation dataset...")
    # Downsample validation set for faster processing if needed
    if args.val_sample_size > 0:
        val_df_sample = val_df.sample(n=args.val_sample_size, random_state=RANDOM_SEED)
    else:
        val_df_sample = val_df

    X_val = val_df_sample[selected_features].values
    y_val = val_df_sample['label'].values

    # Compute validation similarities using Statevector simulation (extremely fast)
    val_similarities = []
    for x in X_val:
        f_norm = calculate_statevector_fidelity(x, normal_prototype)
        f_att = calculate_statevector_fidelity(x, attack_prototype)
        val_similarities.append(f_att - f_norm)

    val_similarities = np.array(val_similarities)

    # Grid search for best threshold on Validation Weighted Youden's J Statistic
    thresholds = np.linspace(-1.0, 1.0, 201)
    best_j = -2.0
    best_threshold = 0.0
    j_scores = []

    for t in thresholds:
        preds = (val_similarities > t).astype(int)
        rec = recall_score(y_val, preds, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y_val, preds).ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        # Standard Youden's J statistic: Recall - FPR (balances sensitivity and specificity equally)
        j = rec - fpr
        j_scores.append(j)
        if j > best_j:
            best_j = j
            best_threshold = t

    logger.info("Adaptive threshold tuning complete. Best Threshold: %.4f | Val Youden's J: %.4f", best_threshold, best_j)

    artifacts_dir = os.path.join(base_dir, "artifacts") if os.path.isdir(os.path.join(base_dir, "artifacts")) else base_dir
    plots_dir = os.path.join(base_dir, "plots") if os.path.isdir(os.path.join(base_dir, "plots")) else base_dir

    # Save artifacts for real-time adapter
    art_dir = "artifacts" if os.path.isdir("artifacts") else "."
    plots_dir = "plots" if os.path.isdir("plots") else "."

    artifacts = {
        "normal_prototype": normal_prototype.tolist(),
        "attack_prototype": attack_prototype.tolist(),
        "threshold": float(best_threshold),
        "selected_features": selected_features
    }
    with open(os.path.join(artifacts_dir, "quantum_artifacts.json"), "w") as f:
        json.dump(artifacts, f)

    # Plot threshold tuning curve
    plt.figure(figsize=(10, 5))
    plt.plot(thresholds, j_scores, color='#ff006e', linewidth=2, label="Youden's J")
    plt.axvline(best_threshold, color='black', linestyle='--', label=f'Optimal Threshold ({best_threshold:.4f})')
    plt.title('Threshold Optimization on Validation Dataset')
    plt.xlabel('Similarity Score Threshold')
    plt.ylabel("Youden's J Statistic")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(plots_dir, "qml_threshold_tuning.png"))
    plt.close()

    # --- EVALUATION ON TEST SET ---
    # Downsample test set for faster simulation
    if args.test_sample_size > 0:
        test_df_sample = test_df.sample(n=args.test_sample_size, random_state=RANDOM_SEED)
    else:
        test_df_sample = test_df

    X_test = test_df_sample[selected_features].values
    y_test = test_df_sample['label'].values

    logger.info("Evaluating on %d test samples using Ideal and Noisy simulators...", len(X_test))

    # Initialize simulators
    sim_ideal = AerSimulator() if AER_AVAILABLE else None
    sim_noisy = get_noisy_simulator() if AER_AVAILABLE else None

    similarities_ideal = []
    similarities_noisy = []

    start_ideal = time.time()
    for x in X_test:
        f_norm = run_fidelity_estimation(x, normal_prototype, sim_ideal, args.shots)
        f_att = run_fidelity_estimation(x, attack_prototype, sim_ideal, args.shots)
        similarities_ideal.append(f_att - f_norm)
    inf_time_ideal = (time.time() - start_ideal) / len(X_test)

    start_noisy = time.time()
    for x in X_test:
        f_norm = run_fidelity_estimation(x, normal_prototype, sim_noisy, args.shots)
        f_att = run_fidelity_estimation(x, attack_prototype, sim_noisy, args.shots)
        similarities_noisy.append(f_att - f_norm)
    inf_time_noisy = (time.time() - start_noisy) / len(X_test)

    similarities_ideal = np.array(similarities_ideal)
    similarities_noisy = np.array(similarities_noisy)

    # Predictions
    preds_ideal = (similarities_ideal > best_threshold).astype(int)
    preds_noisy = (similarities_noisy > best_threshold).astype(int)

    # Evaluate metrics
    def calculate_metrics(y_true, y_pred, scores, time_taken, name):
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_true, scores)
        mcc = matthews_corrcoef(y_true, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        
        return {
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1,
            "ROC-AUC": roc_auc,
            "FPR": fpr,
            "FNR": fnr,
            "MCC": mcc,
            "Training Time (s)": 0.0,  # Prototype matching has no training loop time
            "Inference Time (s/sample)": time_taken
        }

    metrics_ideal = calculate_metrics(y_test, preds_ideal, similarities_ideal, inf_time_ideal, "Quantum Similarity (Ideal)")
    metrics_noisy = calculate_metrics(y_test, preds_noisy, similarities_noisy, inf_time_noisy, "Quantum Similarity (Noisy)")

    # Save to metrics log
    metrics_qml = pd.DataFrame([metrics_ideal, metrics_noisy])
    metrics_path = os.path.join(artifacts_dir, "quantum_metrics.csv")
    metrics_qml.to_csv(metrics_path, index=False)
    
    # Combined Metrics Output
    logger.info("Quantum Similarity Engine Results:")
    print(metrics_qml.to_string(index=False))

    # --- PLOT QUANTUM GRAPHS ---
    
    # 1. Similarity Distribution
    plt.figure(figsize=(10, 6))
    plt.hist(similarities_ideal[y_test == 0], bins=15, alpha=0.6, label='Benign (Ideal)', color='#06d6a0')
    plt.hist(similarities_ideal[y_test == 1], bins=15, alpha=0.6, label='Malicious (Ideal)', color='#ef476f')
    plt.axvline(best_threshold, color='black', linestyle='--', linewidth=2, label=f'Threshold ({best_threshold:.4f})')
    plt.title('Quantum Similarity Score Distribution')
    plt.xlabel('Similarity Score (Attack Fidelity - Normal Fidelity)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(os.path.join(plots_dir, "qml_similarity_distribution.png"))
    plt.close()

    # 2. Confusion Matrix Comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    cm_ideal = confusion_matrix(y_test, preds_ideal)
    cm_noisy = confusion_matrix(y_test, preds_noisy)

    for idx, (cm, title) in enumerate(zip([cm_ideal, cm_noisy], ["Ideal Simulator", "Noisy Simulator"])):
        ax = axes[idx]
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Purples)
        ax.set_title(f"{title} Confusion Matrix")
        fig.colorbar(im, ax=ax)
        classes = ['Benign', 'Malicious']
        tick_marks = np.arange(len(classes))
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(classes)
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(classes)
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        ax.set_ylabel('Actual Label')
        ax.set_xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "qml_confusion_matrix.png"))
    plt.close()

    # 3. ROC Curve
    plt.figure(figsize=(8, 6))
    fpr_i, tpr_i, _ = roc_curve(y_test, similarities_ideal)
    fpr_n, tpr_n, _ = roc_curve(y_test, similarities_noisy)
    plt.plot(fpr_i, tpr_i, color='#118ab2', label=f'Ideal Simulator (AUC = {roc_auc_score(y_test, similarities_ideal):.4f})')
    plt.plot(fpr_n, tpr_n, color='#ffd166', label=f'Noisy Simulator (AUC = {roc_auc_score(y_test, similarities_noisy):.4f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves - Quantum Similarity Engine')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(os.path.join(plots_dir, "qml_roc_curve.png"))
    plt.close()

    # 4. Final Models Comparison Graph
    classical_metrics_candidates = [
        os.path.join(artifacts_dir, "classical_metrics.csv"),
        os.path.join(base_dir, "classical_metrics.csv"),
        "artifacts/classical_metrics.csv",
        "classical_metrics.csv"
    ]
    classical_metrics_path = next((p for p in classical_metrics_candidates if os.path.exists(p)), None)
    if classical_metrics_path:
        classical_df = pd.read_csv(classical_metrics_path)
        combined_df = pd.concat([classical_df, metrics_qml], ignore_index=True)
        
        plt.figure(figsize=(12, 6))
        x_pts = np.arange(len(combined_df))
        b_width = 0.35
        
        plt.bar(x_pts - b_width/2, combined_df['Accuracy'] * 100, b_width, label='Accuracy (%)', color='#073b4c')
        plt.bar(x_pts + b_width/2, combined_df['F1-Score'] * 100, b_width, label='F1-Score (%)', color='#06d6a0')
        plt.title('Hybrid System Performance Comparison')
        plt.xticks(x_pts, combined_df['Model'], rotation=15)
        plt.ylabel('Percentage')
        plt.ylim(30, 105)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "final_model_comparison.png"))
        plt.close()
        logger.info("Saved combined performance plot to %s", os.path.join(plots_dir, "final_model_comparison.png"))

if __name__ == '__main__':
    main()
