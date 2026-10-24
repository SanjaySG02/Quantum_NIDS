import os
import sys
import time
import json
import logging
import argparse
import numpy as np

# Qiskit imports
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector, state_fidelity
try:
    from qiskit_aer import AerSimulator
    AER_AVAILABLE = True
except ImportError:
    AerSimulator = None
    AER_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("LiveSnifferNIDS")

# Suppress verbose external package logs
logging.getLogger('qiskit').setLevel(logging.WARNING)
if AER_AVAILABLE:
    logging.getLogger('qiskit_aer').setLevel(logging.WARNING)
logging.getLogger('stevedore').setLevel(logging.WARNING)

try:
    from scapy.all import sniff, IP, TCP, UDP, ICMP
except ImportError:
    logger.error("Scapy is not installed. Please run: pip install scapy")
    sys.exit(1)

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

def calculate_statevector_fidelity(x: np.ndarray, p: np.ndarray) -> float:
    """
    Computes analytical statevector fidelity directly using Qiskit quantum_info.
    """
    num_qubits = len(x)
    qc_x = QuantumCircuit(num_qubits)
    for i in range(num_qubits):
        qc_x.ry(x[i], i)
    for i in range(num_qubits):
        qc_x.h(i)
    for i in range(num_qubits - 1):
        qc_x.cx(i, i+1)
    sv_x = Statevector.from_instruction(qc_x)

    qc_p = QuantumCircuit(num_qubits)
    for i in range(num_qubits):
        qc_p.ry(p[i], i)
    for i in range(num_qubits):
        qc_p.h(i)
    for i in range(num_qubits - 1):
        qc_p.cx(i, i+1)
    sv_p = Statevector.from_instruction(qc_p)

    return float(state_fidelity(sv_x, sv_p))

def run_fidelity_estimation(x: np.ndarray, p: np.ndarray, simulator=None, shots: int = 1024) -> float:
    """
    Runs the compute-uncompute circuit on the AerSimulator (if available) or analytical Statevector.
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

# Global flow table to maintain states of connection streams
# Key: (src_ip, sport, dst_ip, dport, proto)
# Value: {"history": str, "orig_bytes": int, "id.resp_p": int, "originator_ip": str}
flow_table = {}
packet_counter = 0

def process_live_packet(pkt, simulator, selected_features, vocabularies, scaler_scale, scaler_min, normal_prototype, attack_prototype, threshold):
    """
    Callback function that parses captured packets, updates connection flows,
    runs quantum fidelity matching, and alerts for intrusions.
    """
    global flow_table, packet_counter
    
    # We only care about IP packets
    if not pkt.haslayer(IP):
        return

    ip_layer = pkt[IP]
    src_ip = ip_layer.src
    dst_ip = ip_layer.dst
    
    # Parse protocols and ports
    proto_str = "MISSING"
    sport = 0
    dport = 0
    flags = ""
    payload_len = 0
    
    if pkt.haslayer(TCP):
        proto_str = "tcp"
        sport = pkt[TCP].sport
        dport = pkt[TCP].dport
        flags = str(pkt[TCP].flags)
        if pkt[TCP].payload:
            payload_len = len(pkt[TCP].payload)
    elif pkt.haslayer(UDP):
        proto_str = "udp"
        sport = pkt[UDP].sport
        dport = pkt[UDP].dport
        if pkt[UDP].payload:
            payload_len = len(pkt[UDP].payload)
    elif pkt.haslayer(ICMP):
        proto_str = "icmp"
        
    else:
        # Ignore non-TCP/UDP/ICMP packets
        return
        
    # Define connection flow key
    # Ensure bidirectional flows map to the same key by sorting IPs/ports
    if src_ip < dst_ip or (src_ip == dst_ip and sport <= dport):
        flow_key = (src_ip, sport, dst_ip, dport, proto_str)
        direction = "forward"  # Originator -> Responder
    else:
        flow_key = (dst_ip, dport, src_ip, sport, proto_str)
        direction = "reverse"  # Responder -> Originator

    # Calculate headers length
    ip_hdr_len = pkt[IP].ihl * 4 if hasattr(pkt[IP], 'ihl') else 20
    l4_hdr_len = 0
    if pkt.haslayer(TCP):
        l4_hdr_len = pkt[TCP].dataofs * 4 if hasattr(pkt[TCP], 'dataofs') else 20
    elif pkt.haslayer(UDP):
        l4_hdr_len = 8
    total_hdr_len = ip_hdr_len + l4_hdr_len

    # Initialize flow if not present
    if flow_key not in flow_table:
        flow_table[flow_key] = {
            "times": [],
            "fwd_times": [],
            "bwd_times": [],
            "pkt_lens": [],
            "fwd_lens": [],
            "bwd_lens": [],
            "fwd_header_len": 0,
            "bwd_header_len": 0,
            "tot_fwd_pkts": 0,
            "tot_bwd_pkts": 0,
            "init_bwd_win_bytes": None,
            "fin_flag_cnt": 0,
            "syn_flag_cnt": 0,
            "rst_flag_cnt": 0,
            "ack_flag_cnt": 0,
            "fwd_act_data_pkts": 0,
            "originator_ip": src_ip
        }
        
    flow = flow_table[flow_key]
    
    # Update flow features
    pkt_time = float(pkt.time) if hasattr(pkt, 'time') else time.time()
    flow["times"].append(pkt_time)
    flow["pkt_lens"].append(len(pkt))

    if pkt.haslayer(TCP):
        flags = str(pkt[TCP].flags)
        if "F" in flags: flow["fin_flag_cnt"] += 1
        if "S" in flags: flow["syn_flag_cnt"] += 1
        if "R" in flags: flow["rst_flag_cnt"] += 1
        if "A" in flags: flow["ack_flag_cnt"] += 1

    if direction == "forward":
        flow["fwd_times"].append(pkt_time)
        flow["fwd_lens"].append(len(pkt))
        flow["fwd_header_len"] += total_hdr_len
        flow["tot_fwd_pkts"] += 1
        if payload_len > 0:
            flow["fwd_act_data_pkts"] += 1
    else:
        flow["bwd_times"].append(pkt_time)
        flow["bwd_lens"].append(len(pkt))
        flow["bwd_header_len"] += total_hdr_len
        flow["tot_bwd_pkts"] += 1
        if flow["init_bwd_win_bytes"] is None and pkt.haslayer(TCP):
            flow["init_bwd_win_bytes"] = float(pkt[TCP].window)

    # Compute NetFlow stats for features extraction
    times = flow["times"]
    fwd_times = flow["fwd_times"]
    bwd_times = flow["bwd_times"]
    pkt_lens = flow["pkt_lens"]
    fwd_lens = flow["fwd_lens"]
    bwd_lens = flow["bwd_lens"]
    
    flow_duration = times[-1] - times[0] if len(times) > 1 else 0.0
    flow_iat = np.diff(times) if len(times) > 1 else np.array([])
    fwd_iat = np.diff(fwd_times) if len(fwd_times) > 1 else np.array([])
    bwd_iat = np.diff(bwd_times) if len(bwd_times) > 1 else np.array([])
    
    flow_pkts_s = len(times) / flow_duration if flow_duration > 0 else 0.0
    fwd_pkts_s = len(fwd_times) / flow_duration if flow_duration > 0 else 0.0
    bwd_pkts_s = len(bwd_times) / flow_duration if flow_duration > 0 else 0.0
    flow_byts_s = sum(pkt_lens) / flow_duration if flow_duration > 0 else 0.0
    
    computed_flow = {
        "Flow Duration": flow_duration,
        "Tot Fwd Pkts": flow["tot_fwd_pkts"],
        "Tot Bwd Pkts": flow["tot_bwd_pkts"],
        "TotLen Fwd Pkts": sum(fwd_lens),
        "TotLen Bwd Pkts": sum(bwd_lens),
        "Fwd Pkt Len Max": float(max(fwd_lens)) if fwd_lens else 0.0,
        "Fwd Pkt Len Min": float(min(fwd_lens)) if fwd_lens else 0.0,
        "Fwd Pkt Len Mean": float(np.mean(fwd_lens)) if fwd_lens else 0.0,
        "Fwd Pkt Len Std": float(np.std(fwd_lens)) if fwd_lens else 0.0,
        "Bwd Pkt Len Max": float(max(bwd_lens)) if bwd_lens else 0.0,
        "Bwd Pkt Len Min": float(min(bwd_lens)) if bwd_lens else 0.0,
        "Bwd Pkt Len Mean": float(np.mean(bwd_lens)) if bwd_lens else 0.0,
        "Bwd Pkt Len Std": float(np.std(bwd_lens)) if bwd_lens else 0.0,
        "Flow Byts/s": float(flow_byts_s),
        "Flow Pkts/s": float(flow_pkts_s),
        "Flow IAT Mean": float(np.mean(flow_iat)) if len(flow_iat) > 0 else 0.0,
        "Flow IAT Std": float(np.std(flow_iat)) if len(flow_iat) > 0 else 0.0,
        "Flow IAT Max": float(np.max(flow_iat)) if len(flow_iat) > 0 else 0.0,
        "Flow IAT Min": float(np.min(flow_iat)) if len(flow_iat) > 0 else 0.0,
        "Fwd IAT Tot": float(fwd_times[-1] - fwd_times[0]) if len(fwd_times) > 1 else 0.0,
        "Fwd IAT Mean": float(np.mean(fwd_iat)) if len(fwd_iat) > 0 else 0.0,
        "Fwd IAT Std": float(np.std(fwd_iat)) if len(fwd_iat) > 0 else 0.0,
        "Fwd IAT Max": float(np.max(fwd_iat)) if len(fwd_iat) > 0 else 0.0,
        "Fwd IAT Min": float(np.min(fwd_iat)) if len(fwd_iat) > 0 else 0.0,
        "Bwd IAT Tot": float(bwd_times[-1] - bwd_times[0]) if len(bwd_times) > 1 else 0.0,
        "Bwd IAT Mean": float(np.mean(bwd_iat)) if len(bwd_iat) > 0 else 0.0,
        "Bwd IAT Std": float(np.std(bwd_iat)) if len(bwd_iat) > 0 else 0.0,
        "Bwd IAT Max": float(np.max(bwd_iat)) if len(bwd_iat) > 0 else 0.0,
        "Bwd IAT Min": float(np.min(bwd_iat)) if len(bwd_iat) > 0 else 0.0,
        "Bwd PSH Flags": 0.0,
        "Fwd Header Len": float(flow["fwd_header_len"]),
        "Bwd Header Len": float(flow["bwd_header_len"]),
        "Fwd Pkts/s": float(fwd_pkts_s),
        "Bwd Pkts/s": float(bwd_pkts_s),
        "Pkt Len Min": float(min(pkt_lens)) if pkt_lens else 0.0,
        "Pkt Len Max": float(max(pkt_lens)) if pkt_lens else 0.0,
        "Pkt Len Mean": float(np.mean(pkt_lens)) if pkt_lens else 0.0,
        "Pkt Len Std": float(np.std(pkt_lens)) if pkt_lens else 0.0,
        "Pkt Len Var": float(np.var(pkt_lens)) if pkt_lens else 0.0,
        "FIN Flag Cnt": float(flow["fin_flag_cnt"]),
        "SYN Flag Cnt": float(flow["syn_flag_cnt"]),
        "RST Flag Cnt": float(flow["rst_flag_cnt"]),
        "ACK Flag Cnt": float(flow["ack_flag_cnt"]),
        "Down/Up Ratio": float(flow["tot_bwd_pkts"] / flow["tot_fwd_pkts"]) if flow["tot_fwd_pkts"] > 0 else 0.0,
        "Pkt Size Avg": float(np.mean(pkt_lens)) if pkt_lens else 0.0,
        "Fwd Seg Size Avg": float(np.mean(fwd_lens)) if fwd_lens else 0.0,
        "Bwd Seg Size Avg": float(np.mean(bwd_lens)) if bwd_lens else 0.0,
        "Init Bwd Win Byts": float(flow["init_bwd_win_bytes"]) if flow["init_bwd_win_bytes"] is not None else -1.0,
        "Fwd Act Data Pkts": float(flow["fwd_act_data_pkts"]),
        "Active Mean": 0.0,
        "Active Std": 0.0,
        "Active Max": 0.0,
        "Active Min": 0.0,
        "Idle Mean": 0.0,
        "Idle Std": 0.0,
        "Idle Max": 0.0,
        "Idle Min": 0.0
    }

    # Extract 4 raw features
    raw_values = []
    for feat in selected_features:
        val = computed_flow.get(feat, 0.0)
        raw_values.append(float(val))
        
    # Scale to [0, pi]
    scaled_values = []
    for i in range(len(selected_features)):
        scaled_v = raw_values[i] * scaler_scale[i] + scaler_min[i]
        # Keep inside quantum phase bounds [0, pi]
        scaled_v = np.clip(scaled_v, 0.0, np.pi)
        scaled_values.append(scaled_v)
        
    scaled_values = np.array(scaled_values)
    
    # Run Quantum Similarity calculation
    packet_counter += 1
    start_time = time.time()
    
    f_norm = run_fidelity_estimation(scaled_values, normal_prototype, simulator, shots=1024)
    f_att = run_fidelity_estimation(scaled_values, attack_prototype, simulator, shots=1024)
    
    similarity_score = f_att - f_norm
    prediction = "MALICIOUS" if similarity_score > threshold else "BENIGN"
    
    # Confidence score calculation
    if (f_norm + f_att) > 0:
        confidence = (max(f_norm, f_att) / (f_norm + f_att)) * 100
    else:
        confidence = 50.0
        
    latency_ms = (time.time() - start_time) * 1000
    
    # Output formatting
    color_start = "\033[91m" if prediction == "MALICIOUS" else "\033[92m"
    color_end = "\033[0m"
    prediction_colored = f"{color_start}{prediction:<12}{color_end}"
    
    line_fmt = "{:<10} | {:<12} | {:<15} | {:<5} | {:<15} | {:<5} | {:<10} | {:<10} | {:<10}"
    print(line_fmt.format(
        packet_counter,
        prediction_colored,
        src_ip,
        sport,
        dst_ip,
        dport,
        f"{similarity_score:+.4f}",
        f"{confidence:.1f}%",
        f"{latency_ms:.1f}ms"
    ))

def main():
    # Load quantum configurations
    base_dir = os.path.dirname(os.path.abspath(__file__))
    artifacts_candidates = [
        os.path.join(base_dir, "artifacts", "quantum_artifacts.json"),
        os.path.join(base_dir, "quantum_artifacts.json"),
        "artifacts/quantum_artifacts.json",
        "quantum_artifacts.json"
    ]
    meta_candidates = [
        os.path.join(base_dir, "artifacts", "preprocessing_meta.json"),
        os.path.join(base_dir, "preprocessing_meta.json"),
        "artifacts/preprocessing_meta.json",
        "preprocessing_meta.json"
    ]
    artifacts_path = next((p for p in artifacts_candidates if os.path.exists(p)), "artifacts/quantum_artifacts.json")
    meta_path = next((p for p in meta_candidates if os.path.exists(p)), "artifacts/preprocessing_meta.json")
    
    if not (os.path.exists(artifacts_path) and os.path.exists(meta_path)):
        logger.error("Missing configuration files. Please run classical_nids.py and quantum_nids.py first.")
        sys.exit(1)
        
    logger.info("Loading preprocessing parameters from %s...", meta_path)
    with open(meta_path, "r") as f:
        meta = json.load(f)
        
    selected_features = meta["selected_features"]
    vocabularies = meta["vocabularies"]
    scaler_scale = meta["scaler_scale"]
    scaler_min = meta["scaler_min"]
    
    logger.info("Loading quantum prototypes from %s...", artifacts_path)
    with open(artifacts_path, "r") as f:
        artifacts = json.load(f)
        
    normal_prototype = np.array(artifacts["normal_prototype"])
    attack_prototype = np.array(artifacts["attack_prototype"])
    threshold = artifacts["threshold"]
    
    simulator = AerSimulator() if AER_AVAILABLE else None
    if AER_AVAILABLE:
        logger.info("Using Qiskit AerSimulator for shot-based execution.")
    else:
        logger.info("AerSimulator unavailable on this environment; using Qiskit Statevector engine (Exact & ultra-fast).")
    
    print("\n" + "="*110)
    print("                 LIVE INTRUSION DETECTION SYSTEM: HYBRID QUANTUM STATE FIDELITY ENGINE")
    print("="*110)
    print("Capturing live packets on active network interface cards.")
    print("Press Ctrl+C to stop the sniffer stream.")
    print("="*110)
    
    line_fmt = "{:<10} | {:<12} | {:<15} | {:<5} | {:<15} | {:<5} | {:<10} | {:<10} | {:<10}"
    print(line_fmt.format("Packet #", "Prediction", "Source IP", "Port", "Dest IP", "Port", "Sim Score", "Confidence", "Latency"))
    print("-" * 115)
    
    # Callback wrapper to inject parameters
    def sniffer_callback(pkt):
        process_live_packet(
            pkt, simulator, selected_features, vocabularies, 
            scaler_scale, scaler_min, normal_prototype, attack_prototype, threshold
        )
        
    # Start Scapy Sniffer
    try:
        sniff(prn=sniffer_callback, store=0)
    except (PermissionError, RuntimeError) as e:
        logger.error("\n[EXECUTION ERROR] Live packet sniffing failed.")
        print("\n" + "="*90)
        print("To run the Live NIDS Sniffer on Windows, please ensure:")
        print("1. Npcap is installed:")
        print("   - Scapy requires Npcap to capture raw Ethernet frames on Windows.")
        print("   - Download and install Npcap from: https://npcap.com/")
        print("   - Make sure to enable 'Install Npcap in WinPcap API-compatible mode' during setup.")
        print("2. Run as Administrator:")
        print("   - Search for 'PowerShell' or 'Command Prompt' in the Windows Start menu.")
        print("   - Right-click and choose 'Run as Administrator'.")
        print("   - Navigate to the project directory and execute: python realtime_sniffer.py")
        print("="*90)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n" + "="*110)
        print("Live Quantum NIDS Sniffer terminated.")
        print("="*110)

if __name__ == '__main__':
    main()
