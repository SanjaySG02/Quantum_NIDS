#!/usr/bin/env python3
"""
Benign Client Simulator for Quantum NIDS Demo.
Simulates a legitimate user making a normal single-port connection (e.g. HTTP/service request).
"""

import sys
import time
import socket
import argparse
from datetime import datetime

GREEN = "\033[92m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"

def main():
    parser = argparse.ArgumentParser(description="Benign Client Simulator")
    parser.add_argument("--target", type=str, default="127.0.0.1", help="Target IP address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Target Port (default: 8000)")
    args = parser.parse_args()

    print(f"\n{GREEN}{BOLD}" + "="*70)
    print("      LEGITIMATE USER SIMULATION: BENIGN SINGLE-SERVICE REQUEST")
    print("="*70 + f"{RESET}")
    print(f"{CYAN}[*] Target Host : {args.target}{RESET}")
    print(f"{CYAN}[*] Target Port : {args.port} (Single Service Access){RESET}")
    print(f"{CYAN}[*] Request Type: Standard TCP Service Handshake{RESET}")
    print(f"{CYAN}[*] Timestamp   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{RESET}\n")

    try:
        t0 = time.time()
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(2.0)
        s.connect((args.target, args.port))
        
        # Send a normal benign payload (HTTP GET request)
        payload = b"GET /index.html HTTP/1.1\r\nHost: safe.local\r\n\r\n"
        s.sendall(payload)
        latency_ms = (time.time() - t0) * 1000.0
        
        print(f"{GREEN}[+] Connection Established Successfully! Latency: {latency_ms:.1f}ms{RESET}")
        print(f"{GREEN}[+] Legitimate service request sent.{RESET}")
        time.sleep(0.5)
        s.close()
        
    except ConnectionRefusedError:
        print(f"\033[91m[-] Connection Refused: Port {args.port} is not listening or is blocked by Firewall.\033[0m")
    except Exception as e:
        print(f"\033[91m[-] Connection Error: {e}\033[0m")

    print("\n" + "="*70)
    print(f"{BOLD}SUMMARY:{RESET} Normal traffic sent. Check Defender terminal for {GREEN}BENIGN{RESET} classification.")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
