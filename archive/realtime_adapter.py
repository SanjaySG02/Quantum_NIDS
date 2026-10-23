import os
import sys
import time
import json
import logging
import numpy as np
import pandas as pd

# Qiskit imports
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("RealtimeNIDS")

# Suppress verbose external package logs
logging.getLogger('qiskit').setLevel(logging.WARNING)
logging.getLogger('qiskit_aer').setLevel(logging.WARNING)
logging.getLogger('stevedore').setLevel(logging.WARNING)

def construct_fidelity_circuit(x: np.ndarray, p: np.ndarray) -> QuantumCircuit:
    """
    Builds the Compute-Uncompute fidelity circuit between sample state x and prototype p.
    """
    num_qubits = len(x)
    qc = QuantumCircuit(num_qubits)
    
    # 1. Encode sample x
    for i in range(num_qubits):
        qc.ry(x[i], i)
    for i in range(num_qubits):
        qc.h(i)
    for i in range(num_qubits - 1):
        qc.cx(i, i+1)
        
    # 2. Uncompute prototype p
    for i in range(num_qubits - 2, -1, -1):
        qc.cx(i, i+1)
    for i in range(num_qubits):
        qc.h(i)
    for i in range(num_qubits):
        qc.ry(-p[i], i)
        
    qc.measure_all()
    return qc

def run_fidelity_estimation(x: np.ndarray, p: np.ndarray, simulator: AerSimulator, shots: int = 1024) -> float:
    """
    Runs the compute-uncompute circuit on the AerSimulator.
    """
    qc = construct_fidelity_circuit(x, p)
    transpiled_qc = transpile(qc, simulator)
    job = simulator.run(transpiled_qc, shots=shots)
    counts = job.result().get_counts(transpiled_qc)
    
    zero_state = '0' * len(x)
    return counts.get(zero_state, 0) / shots

def main():
    # Load quantum artifacts
    artifacts_path = "quantum_artifacts.json"
    test_data_path = "test_processed.csv"
    
    if not (os.path.exists(artifacts_path) and os.path.exists(test_data_path)):
        logger.error("Missing required configuration or dataset. Please run classical_nids.py and quantum_nids.py first.")
        sys.exit(1)
        
    logger.info("Loading QML artifacts from %s...", artifacts_path)
    with open(artifacts_path, "r") as f:
        artifacts = json.load(f)
        
    normal_prototype = np.array(artifacts["normal_prototype"])
    attack_prototype = np.array(artifacts["attack_prototype"])
    threshold = artifacts["threshold"]
    selected_features = artifacts["selected_features"]
    
    logger.info("Selected Features for Real-Time Detection: %s", selected_features)
    logger.info("Decision Boundary Threshold: %.4f", threshold)

    logger.info("Loading testing dataset from %s...", test_data_path)
    test_df = pd.read_csv(test_data_path)
    
    # Initialize Ideal simulator
    simulator = AerSimulator()
    
    print("\n" + "="*80)
    print("      HYBRID QUANTUM NIDS REAL-TIME PACKET INTRUSION SIMULATOR")
    print("="*80)
    print("Press Ctrl+C to terminate the simulator stream at any time.")
    print("="*80)
    
    # Formatted Header
    header_fmt = "{:<10} | {:<12} | {:<10} | {:<10} | {:<10} | {:<10} | {:<10}"
    print(header_fmt.format("Packet #", "Prediction", "Normal Fid", "Attack Fid", "Sim Score", "Confidence", "Latency"))
    print("-" * 86)
    
    packet_count = 0
    shots = 1024
    
    try:
        for idx, row in test_df.iterrows():
            packet_count += 1
            x = row[selected_features].values
            actual_label = int(row['label'])
            
            # Start timer
            start_time = time.time()
            
            # Compute State Fidelity with prototypes
            f_norm = run_fidelity_estimation(x, normal_prototype, simulator, shots)
            f_att = run_fidelity_estimation(x, attack_prototype, simulator, shots)
            
            # Calculate Similarity Score & Decision
            similarity_score = f_att - f_norm
            prediction = "MALICIOUS" if similarity_score > threshold else "BENIGN"
            
            # Calculate Confidence Score: max(Fn, Fa) / (Fn + Fa)
            if (f_norm + f_att) > 0:
                confidence = (max(f_norm, f_att) / (f_norm + f_att)) * 100
            else:
                confidence = 50.0
                
            latency_ms = (time.time() - start_time) * 1000
            
            # Format outputs
            sim_str = f"{similarity_score:+.4f}"
            conf_str = f"{confidence:.1f}%"
            latency_str = f"{latency_ms:.1f}ms"
            
            # Add color tags based on prediction (using ANSI escape sequences for premium look)
            color_start = "\033[91m" if prediction == "MALICIOUS" else "\033[92m"
            color_end = "\033[0m"
            prediction_colored = f"{color_start}{prediction:<12}{color_end}"
            
            # Print row
            print(header_fmt.format(
                packet_count,
                prediction_colored,
                f"{f_norm:.4f}",
                f"{f_att:.4f}",
                sim_str,
                conf_str,
                latency_str
            ))
            
            # Wait 1 second
            time.sleep(1.0)
            
    except KeyboardInterrupt:
        print("\n" + "="*80)
        print("Real-Time Intrusion Detection Simulator stopped by user.")
        print("="*80)
        
if __name__ == '__main__':
    main()
