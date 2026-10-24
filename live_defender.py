#!/usr/bin/env python3
"""
Hybrid Quantum Network Intrusion Detection System (HQNIDS)
Live Active Defense & Port Scan Monitor.

Features:
- Multi-port socket sentry
- Quantum compute-uncompute state fidelity estimation (Qiskit Aer)
- Autonomous active defense: Dynamic Windows Firewall isolation + Workstation Lockdown GUI
"""

import os
import sys
import time
import json
import socket
import select
import logging
import threading
import subprocess
from datetime import datetime
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

# Logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("QuantumGuardian")
logging.getLogger('qiskit').setLevel(logging.WARNING)
if AER_AVAILABLE:
    logging.getLogger('qiskit_aer').setLevel(logging.WARNING)
logging.getLogger('stevedore').setLevel(logging.WARNING)

# ANSI formatting
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"

# =====================================================================
# 1. QUANTUM FIDELITY ENGINE (Compute-Uncompute Circuit)
# =====================================================================

def construct_fidelity_circuit(x: np.ndarray, p: np.ndarray) -> QuantumCircuit:
    """
    Constructs the compute-uncompute quantum fidelity estimation circuit:
    |0...0> --[ U(x) ]--[ U(p)^dagger ]-- Measure
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
        
    # 2. Uncompute prototype p: U(p)^dagger
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

# =====================================================================
# 2. ACTIVE DEFENSE & LOCKDOWN MODULE
# =====================================================================

def isolate_attacker_ip(attacker_ip: str) -> bool:
    """
    Adds an active inbound blocking rule in Windows Firewall to isolate attacker.
    """
    if attacker_ip in ["127.0.0.1", "localhost", "::1"]:
        logger.info("[ACTIVE DEFENSE] Attacker is on loopback interface (%s). Simulating firewall drop.", attacker_ip)
        return True
        
    rule_name = f"Quantum_NIDS_Block_{attacker_ip.replace('.', '_')}"
    cmd = [
        "netsh", "advfirewall", "firewall", "add", "rule",
        f"name={rule_name}",
        "dir=in",
        "action=block",
        f"remoteip={attacker_ip}"
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if res.returncode == 0:
            logger.info("[ACTIVE DEFENSE] Windows Firewall rule created: Blocked %s", attacker_ip)
            return True
        else:
            logger.warning("[ACTIVE DEFENSE] Firewall rule could not be added (Requires Administrator): %s", res.stderr.strip())
            return False
    except Exception as e:
        logger.error("[ACTIVE DEFENSE] Error invoking Windows Firewall: %s", e)
        return False

def lock_workstation():
    """
    Calls the native Windows LockWorkStation API.
    """
    try:
        logger.warning("[LOCKDOWN] Executing Windows LockWorkStation...")
        subprocess.run(["rundll32.exe", "user32.dll,LockWorkStation"], check=False)
    except Exception as e:
        logger.error("[LOCKDOWN] Failed to lock workstation: %s", e)

def launch_containment_gui(threat_data: dict, auto_lock: bool = True, countdown_sec: int = 7):
    """
    Spawns the high-tech alert HUD window as an independent process
    so it appears reliably on top every single time without thread deadlocks.
    """
    threat_data["countdown"] = countdown_sec
    base_dir = os.path.dirname(os.path.abspath(__file__))
    hud_candidates = [
        os.path.join(base_dir, "simulators", "alert_hud.py"),
        os.path.join(base_dir, "alert_hud.py"),
        "simulators/alert_hud.py",
        "alert_hud.py"
    ]
    hud_script = next((p for p in hud_candidates if os.path.exists(p)), "alert_hud.py")
    cmd = [sys.executable, hud_script, json.dumps(threat_data)]
    if auto_lock:
        cmd.append("--auto-lock")
    try:
        subprocess.Popen(cmd)
        logger.info("[HUD] Spawned threat containment overlay window.")
    except Exception as e:
        logger.error("[HUD] Failed to spawn alert HUD window: %s", e)
        if auto_lock:
            lock_workstation()

# =====================================================================
# 3. LIVE MULTI-PORT SENTRY & FLOW TRACKER
# =====================================================================

class FlowTracker:
    def __init__(self):
        self.flows = {}  # src_ip -> list of packet times, ports, etc.

    def record_probe(self, src_ip: str, port: int) -> dict:
        now = time.time()
        is_new_burst = False
        if src_ip not in self.flows:
            self.flows[src_ip] = {
                "times": [],
                "ports": set(),
                "count": 0,
                "first_seen": now
            }
            is_new_burst = True
        else:
            flow = self.flows[src_ip]
            if len(flow["times"]) > 0 and (now - flow["times"][-1]) > 1.5:
                # More than 1.5 seconds since last probe: treat as a fresh attack burst!
                flow["times"] = []
                flow["ports"] = set()
                flow["count"] = 0
                is_new_burst = True

        flow = self.flows[src_ip]
        flow["times"].append(now)
        flow["ports"].add(port)
        flow["count"] += 1
        
        # Calculate Flow Features matching our 4 selected features:
        # ["Flow IAT Min", "Init Bwd Win Byts", "Flow IAT Std", "Bwd Header Len"]
        times = flow["times"]
        
        # If the host is probing multiple ports rapidly, it exhibits port-scan kinetics
        if len(flow["ports"]) > 1:
            iats = np.diff(times) if len(times) > 1 else [0.05]
            flow_iat_min = float(np.min(iats)) * 1e6   # Microseconds
            flow_iat_std = float(np.std(iats)) * 1e6   # Microseconds
            init_bwd_win_bytes = 64240.0              # Reconnaissance probe window
            bwd_header_len = 20.0 * flow["count"]
        else:
            # Normal single-port connection (Benign Baseline)
            flow_iat_min = 1000.0                      # Microseconds
            flow_iat_std = 0.0
            init_bwd_win_bytes = 0.0                  # Benign window baseline
            bwd_header_len = 40.0
        
        return {
            "Flow IAT Min": flow_iat_min,
            "Init Bwd Win Byts": init_bwd_win_bytes,
            "Flow IAT Std": flow_iat_std,
            "Bwd Header Len": bwd_header_len,
            "ports_scanned": sorted(list(flow["ports"])),
            "count": flow["count"],
            "is_new_burst": is_new_burst
        }

# =====================================================================
# 4. MAIN DEFENDER LOOP
# =====================================================================

def get_local_ip():
    """Returns the primary active local IP address on Wi-Fi / LAN."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Live Quantum Guardian NIDS")
    parser.add_argument("--no-lock", action="store_true", help="Run without locking Windows screen (for safe testing)")
    parser.add_argument("--countdown", type=int, default=7, help="Countdown seconds before workstation lock (default: 7)")
    args = parser.parse_args()

    auto_lock_enabled = not args.no_lock
    countdown_seconds = args.countdown
    local_ip = get_local_ip()

    print(f"\n{CYAN}{BOLD}" + "="*85)
    print("      HYBRID QUANTUM NIDS - LIVE ACTIVE DEFENSE GUARDIAN")
    print("      Real-Time Quantum State Fidelity & Autonomous Workstation Lockdown")
    print("="*85 + f"{RESET}")
    print(f"{YELLOW}[*] Target Host IP             : {BOLD}{local_ip}{RESET}")
    print(f"{CYAN}[*] Attacker Command (Laptop 2): {BOLD}python port_scan_attacker.py --target {local_ip}{RESET}")
    print(f"{GREEN}[*] Benign Command (Laptop 2)  : {BOLD}python benign_client.py --target {local_ip}{RESET}")
    if not auto_lock_enabled:
        print(f"{YELLOW}[!] SAFE DRY-RUN MODE: Screen lock is DISABLED (--no-lock){RESET}")

    # Load artifacts and metadata
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
        logger.error("Missing configuration files (%s, %s). Please ensure models are trained.", artifacts_path, meta_path)
        sys.exit(1)

    with open(artifacts_path, "r") as f:
        artifacts = json.load(f)
    with open(meta_path, "r") as f:
        meta = json.load(f)

    selected_features = meta["selected_features"]
    scaler_scale = meta["scaler_scale"]
    scaler_min = meta["scaler_min"]
    normal_prototype = np.array(artifacts["normal_prototype"])
    attack_prototype = np.array(artifacts["attack_prototype"])
    threshold = artifacts["threshold"]

    logger.info("Quantum Model Initialized.")
    logger.info("Artifacts loaded from: %s", artifacts_path)
    logger.info("Selected Features: %s", selected_features)
    logger.info("Threshold Boundary: %.4f", threshold)
    logger.info("Normal Prototype State: %s", normal_prototype.round(4).tolist())
    logger.info("Attack Prototype State: %s", attack_prototype.round(4).tolist())

    simulator = AerSimulator() if AER_AVAILABLE else None
    if AER_AVAILABLE:
        logger.info("Using Qiskit AerSimulator for shot-based execution.")
    else:
        logger.info("AerSimulator unavailable on this environment; using Qiskit Statevector engine (Exact & ultra-fast).")
    flow_tracker = FlowTracker()

    # Sentry Port Range to Monitor
    start_port = 8000
    end_port = 8015
    server_sockets = []

    print(f"\n{YELLOW}[*] Deploying Multi-Port Sentry Sensor on Ports {start_port} -> {end_port}...{RESET}")
    for port in range(start_port, end_port + 1):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("0.0.0.0", port))
            s.listen(5)
            s.setblocking(False)
            server_sockets.append(s)
        except Exception as e:
            logger.warning("Could not bind port %d: %s", port, e)

    print(f"{GREEN}[+] Sentry Sensor Active. Listening on {len(server_sockets)} ports.{RESET}")
    print(f"{GREEN}[+] Quantum Aer Simulator Ready. Awaiting Network Probes...{RESET}\n")

    header_fmt = "{:<8} | {:<12} | {:<16} | {:<6} | {:<10} | {:<10} | {:<10} | {:<10} | {:<8}"
    print(header_fmt.format("Event #", "Prediction", "Source IP", "Port", "F_Normal", "F_Attack", "Sim Score", "Confidence", "Latency"))
    print("-" * 105)

    event_counter = 0
    shots = 1024
    already_quarantined = set()

    try:
        while True:
            # Poll sockets for incoming connections
            readable, _, _ = select.select(server_sockets, [], [], 0.5)
            
            for s in readable:
                try:
                    conn, addr = s.accept()
                    src_ip, src_port = addr
                    target_port = s.getsockname()[1]
                    conn.close()
                except Exception:
                    continue

                event_counter += 1
                t0 = time.time()

                # Record probe & extract flow statistics
                flow_data = flow_tracker.record_probe(src_ip, target_port)
                
                # Format 4 raw feature values
                raw_values = [flow_data.get(feat, 0.0) for feat in selected_features]

                # Scale to [0, pi] for Quantum Phase Encoding
                scaled_values = []
                for i in range(len(selected_features)):
                    sv = raw_values[i] * scaler_scale[i] + scaler_min[i]
                    scaled_values.append(np.clip(sv, 0.0, np.pi))
                scaled_values = np.array(scaled_values)

                # Compute Quantum State Overlap Fidelity
                f_norm = run_fidelity_estimation(scaled_values, normal_prototype, simulator, shots=shots)
                f_att = run_fidelity_estimation(scaled_values, attack_prototype, simulator, shots=shots)

                sim_score = f_att - f_norm
                prediction = "MALICIOUS" if sim_score > threshold else "BENIGN"
                
                # Confidence Calculation
                if (f_norm + f_att) > 0:
                    confidence = (max(f_norm, f_att) / (f_norm + f_att)) * 100.0
                else:
                    confidence = 50.0

                latency_ms = (time.time() - t0) * 1000.0

                # Formatted colored output
                if prediction == "MALICIOUS":
                    pred_col = f"{RED}{BOLD}MALICIOUS{RESET}"
                else:
                    pred_col = f"{GREEN}BENIGN   {RESET}"

                print(header_fmt.format(
                    event_counter,
                    pred_col,
                    src_ip,
                    target_port,
                    f"{f_norm:.4f}",
                    f"{f_att:.4f}",
                    f"{sim_score:+.4f}",
                    f"{confidence:.1f}%",
                    f"{latency_ms:.1f}ms"
                ), flush=True)

                # If a fresh burst started, reset quarantine state for this IP
                if flow_data.get("is_new_burst", False):
                    already_quarantined.discard(src_ip)

                # If Malicious and not yet quarantined: Trigger Autonomous Defense
                if prediction == "MALICIOUS" and src_ip not in already_quarantined and flow_data["count"] >= 2:
                    already_quarantined.add(src_ip)
                    print(f"\n{RED}{BOLD}>>> [!] INTRUSION DETECTED! ACTIVATING AUTONOMOUS WORKSTATION LOCKDOWN <<<{RESET}", flush=True)
                    print(f"{RED}[!] Attacker IP: {src_ip} | Scanned Ports: {flow_data['ports_scanned']}{RESET}", flush=True)
                    print(f"{RED}[!] Quantum Attack Fidelity: {f_att:.4f} | Confidence: {confidence:.1f}%{RESET}\n", flush=True)

                    # 1. Firewall Isolation
                    isolate_attacker_ip(src_ip)

                    # 2. Spawn Emergency Containment GUI & Lock Workstation
                    threat_info = {
                        "attacker_ip": src_ip,
                        "ports_scanned": f"{len(flow_data['ports_scanned'])} ports {flow_data['ports_scanned'][:5]}...",
                        "f_att": f_att,
                        "f_norm": f_norm,
                        "sim_score": sim_score,
                        "threshold": threshold,
                        "confidence": confidence,
                        "latency_ms": latency_ms
                    }
                    launch_containment_gui(threat_info, auto_lock=auto_lock_enabled, countdown_sec=countdown_seconds)

    except KeyboardInterrupt:
        print(f"\n{YELLOW}[*] Shutting down Quantum Guardian Sensor...{RESET}")
    finally:
        for s in server_sockets:
            s.close()
        print(f"{GREEN}[+] All sentry sockets closed cleanly.{RESET}\n")

if __name__ == "__main__":
    main()
