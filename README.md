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
├── 1_START_DEFENDER.bat      # 1-Click Desktop Launcher for Defender (Laptop 1)
├── 2_SAFE_TEST.bat           # 1-Click Desktop Launcher for Benign Client (Laptop 2)
├── 3_ATTACK_TEST.bat         # 1-Click Desktop Launcher for Port Scan Attacker (Laptop 2)
├── live_defender.py          # Core: Multi-Port Sentry, Quantum Similarity Engine & Active Defense
├── classical_nids.py         # Classical baseline model trainer & feature selector
├── quantum_nids.py           # Offline Quantum Prototype Trainer & Aer Simulator Benchmark
├── realtime_sniffer.py       # Layer-2 packet sniffer (Npcap adapter)
├── requirements.txt          # Project dependencies
├── README.md                 # Documentation & Presentation Demo Guide
├── simulators/               # Network simulation tools & threat containment GUI
│   ├── benign_client.py      # Legitimate User Single-Service Request Simulator
│   ├── port_scan_attacker.py # Adversarial Multi-Port Reconnaissance Sweep
│   └── alert_hud.py          # Standalone Threat Containment & Lock Workstation GUI
├── artifacts/                # Trained model artifacts, prototypes & performance metrics
│   ├── quantum_artifacts.json
│   ├── preprocessing_meta.json
│   ├── selected_features.json
│   ├── classical_metrics.csv
│   └── quantum_metrics.csv
├── data/                     # Cleaned, split & processed network datasets
│   ├── CTU13_Merged_Traffic.csv
│   ├── train_processed.csv
│   ├── val_processed.csv
│   └── test_processed.csv
├── plots/                    # Generated ROC curves, confusion matrices & comparison charts
└── archive/                  # Archived experimental scripts (GNN, Model Comparison)
```

---

## 3. Installation & Requirements

Ensure you have Python 3.10+ installed:

```bash
git clone https://github.com/SanjaySG02/Quantum_NIDS.git
cd Quantum_NIDS
pip install -r requirements.txt
```

---

## 4. How to Run the Live 2-Laptop Autonomous Defense Demo

You can run the demo either with **1-Click Launchers (Recommended for Demo)** or via **Command Line**:

### Option A: 1-Click Double-Click Launchers (Fastest & Typo-Free)
* **Laptop 1 (Defender)**: Double-click `1_START_DEFENDER.bat` -> Press `1` for Live Lockdown Mode (or `2` for Safe Dry-Run). Note the IP shown.
* **Laptop 2 (Safe Test)**: Double-click `2_SAFE_TEST.bat` -> Enter Laptop 1's IP -> Legitimate traffic sent.
* **Laptop 2 (Attack Test)**: Double-click `3_ATTACK_TEST.bat` -> Press Enter to reuse cached IP -> Reconnaissance sweep launched!

---

### Option B: Command Line

#### Step 1: Launch the Quantum Guardian (Laptop 1 - Target / Defender)
Run the live quantum defender. On startup, it automatically detects and prints your laptop's active IP address:

```bash
# Safe testing mode (shows threat HUD without locking your screen):
python live_defender.py --no-lock

# Live presentation mode (initiates autonomous workstation lock on intrusion):
python live_defender.py
```

#### Step 2: Demonstrate Normal Traffic (Laptop 2 - Legitimate User)
Demonstrates that the quantum engine distinguishes safe traffic and avoids false positives:

```bash
python simulators/benign_client.py --target <LAPTOP_1_IP>
```
* **Result**: Laptop 1 evaluates the quantum state fidelity against the normal prototype ($F_{\text{Normal}} \approx 0.74 > F_{\text{Attack}} \approx 0.44$).
* **Decision**: **`BENIGN`** (Green). No alarm, no lock.

#### Step 3: Launch the Adversarial Attack (Laptop 2 - Hacker / Reconnaissance)
Simulates an aggressive multi-port TCP reconnaissance sweep:

```bash
python simulators/port_scan_attacker.py --target <LAPTOP_1_IP>
```
* **Result**: Laptop 1 computes the quantum state overlap ($F_{\text{Attack}} \approx 0.59 > F_{\text{Normal}} \approx 0.29$).
* **Decision**: **`MALICIOUS`** (Red) triggered in under 200 milliseconds.
* **Autonomous Containment**:
  1. Installs an inbound Windows Firewall drop rule for Laptop 2's IP (`netsh advfirewall`).
  2. Spawns the **Quantum Threat Alert HUD** (`alert_hud.py`).
  3. Executes `LockWorkStation` to immediately secure the physical machine.
