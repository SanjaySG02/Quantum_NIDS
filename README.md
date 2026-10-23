# Hybrid Quantum Network Intrusion Detection System (HQNIDS)

This project implements a complete, modular, and production-quality **Hybrid Quantum Network Intrusion Detection System (HQNIDS)** for IoT networks. Instead of standard black-box classifiers (like VQCs or QSVs), it introduces an explainable **Hybrid Quantum Similarity Engine** based on **Quantum Feature Encoding** and **Quantum State Fidelity**.

---

## 1. Project Overview & Architecture

Modern Network Intrusion Detection Systems (NIDS) struggle with high-dimensional data, latency constraints, and model explainability. HQNIDS resolves these by combining classical machine learning with quantum computation:

1. **Dimensionality Reduction**: A classical Random Forest Classifier selects the top-4 most important features from the raw dataset, resolving the qubit limit bottleneck in NISQ-era quantum hardware.
2. **Feature Mapping**: Features are scaled to the range $[0, \pi]$ and encoded into a 4-qubit quantum state using $RY$ rotation gates, superposition (Hadamard layers), and linear entanglement (CNOT gates).
3. **Quantum Similarity Engine**: Instead of training a variational quantum classifier, we construct two **Quantum Prototype States**:
   * **Normal State ($|\psi_{\text{Normal}}\rangle$)**: Constructed from the class mean feature vector of normal traffic.
   * **Attack State ($|\psi_{\text{Attack}}\rangle$)**: Constructed from the class mean feature vector of malicious traffic.
4. **Fidelity Estimation**: For any incoming network packet $x$, the system builds a **Compute-Uncompute circuit** ($U^\dagger(\text{Prototype}) U(x)$) to estimate the fidelity overlaps $F_{\text{Normal}}$ and $F_{\text{Attack}}$ using `Qiskit Aer` simulators.
5. **Adaptive Threshold Decision**: A similarity score ($S = F_{\text{Attack}} - F_{\text{Normal}}$) is calculated. If the score is higher than the validation-set optimal threshold, the packet is flagged as malicious.

---

## 2. Folder Structure

```text
Quantum_NIDS/
├── CTU13_Merged_Traffic.csv    # Merged CTU-13 IoT Traffic Dataset
├── selected_features.json     # Output: Saved top-K feature names
├── preprocessing_meta.json    # Output: Saved MinMaxScaler bounds and categorical vocabularies
├── train_processed.csv        # Output: Normalized training dataset
├── val_processed.csv          # Output: Normalized validation dataset
├── test_processed.csv         # Output: Normalized testing dataset
├── classical_metrics.csv      # Output: Evaluated classical model metrics
├── quantum_metrics.csv        # Output: Evaluated ideal/noisy quantum metrics
├── quantum_artifacts.json     # Output: Saved prototype coordinates and threshold
├── classical_nids.py          # Script: Classical data prep, splits, RF selection & training
├── quantum_nids.py            # Script: State Fidelity Similarity engine, Ideal/Noisy evaluation
├── train_gnn.py               # Script: PyTorch Graph Convolutional Network (GCN) on k-NN graph
├── compare_models.py          # Script: Comprehensive benchmark (RF, XGBoost, GNN, LSTM)
├── realtime_adapter.py        # Script: Real-time network stream simulation from test CSV
├── realtime_sniffer.py        # Script: Real-time live network packet capture and sniffer
├── requirements.txt           # Project dependencies
└── README.md                  # Documentation: Project overview & execution guide
```

---

## 3. Installation & Requirements

Ensure you have Python installed, then clone the repository and install all dependencies:

```bash
git clone https://github.com/SanjaySG02/Quantum_NIDS.git
cd Quantum_NIDS
pip install -r requirements.txt
```

---

## 4. How to Run

Follow these steps in order to demonstrate the complete integrated HQNIDS:

### Step 1: Execute the Classical Pipeline
This script handles clean splitting, trains the classical models, generates the scaled processed files, and identifies the best 4 features.
```bash
python classical_nids.py
```

### Step 2: Run the Quantum Similarity Engine
Loads the processed CSV splits and builds the prototypes. Optimizes the threshold on the validation split and runs both Ideal and Noisy quantum simulations.
```bash
python quantum_nids.py --val_sample_size 300 --test_sample_size 150
```
*You can customize simulation speeds using `--val_sample_size` and `--test_sample_size`. Set them to `0` to run on the entire dataset.*

### Step 3: Launch the Real-Time Intrusion Simulator (Offline)
Simulates a live router flow by drawing packets from the test split every second, computing real-time quantum similarities, and logging detections.
```bash
python realtime_adapter.py
```
*Press `Ctrl+C` in the console window to stop the stream.*

### Step 4: Launch the Live Packet Capture NIDS Adapter (Real-Time Live Network)
Captures actual packets from your physical network adapter, extracts features, and runs them through the Quantum Similarity Engine.
```bash
python realtime_sniffer.py
```
*Note: Sniffing raw network packets requires administrator privileges. On Windows, open PowerShell as an Administrator and execute the command. Additionally, ensure Npcap is installed on your machine (https://npcap.com/).*
