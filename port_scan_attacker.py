#!/usr/bin/env python3
"""
Port Scan Attack Simulator for Quantum NIDS Demo.
Simulates a multi-port reconnaissance probe against a target IP.
"""

import sys
import time
import socket
import argparse
from datetime import datetime

# ANSI Color codes for attractive terminal output
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"

def print_banner():
    print(f"\n{RED}{BOLD}" + "="*75)
    print("      MALICIOUS ACTOR SIMULATION: RECONNAISSANCE PORT SCANNER")
    print("      Designed for Live Quantum NIDS Defense Demonstration")
    print("="*75 + f"{RESET}")

def scan_target(target_ip: str, start_port: int, end_port: int, delay_s: float):
    print(f"\n{YELLOW}[*] Target Host      : {target_ip}{RESET}")
    print(f"{YELLOW}[*] Port Range       : {start_port} -> {end_port} ({end_port - start_port + 1} ports){RESET}")
    print(f"{YELLOW}[*] Probe Frequency  : {1.0 / delay_s if delay_s > 0 else 999:.1f} packets/second{RESET}")
    print(f"{YELLOW}[*] Scan Profile      : High-Frequency TCP SYN/Connect Sweep{RESET}")
    print(f"{YELLOW}[*] Timestamp         : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{RESET}\n")

    print(f"{CYAN}{'#':<5} | {'Port':<8} | {'Status':<16} | {'Response Time':<15} | {'Target IP'}{RESET}")
    print("-" * 65)

    probe_count = 0
    blocked_count = 0
    open_count = 0

    try:
        for port in range(start_port, end_port + 1):
            probe_count += 1
            t0 = time.time()
            
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(0.6)  # 600ms timeout
            
            try:
                res = s.connect_ex((target_ip, port))
                latency_ms = (time.time() - t0) * 1000
                
                if res == 0:
                    status = f"{GREEN}OPEN (PROBED){RESET}"
                    open_count += 1
                elif res == 10061:  # WSAECONNREFUSED
                    status = f"{RED}REFUSED / CLOSED{RESET}"
                elif res == 10060:  # WSAETIMEDOUT
                    status = f"{RED}{BOLD}BLOCKED (FILTERED){RESET}"
                    blocked_count += 1
                else:
                    status = f"{YELLOW}FILTERED (Code {res}){RESET}"
                    blocked_count += 1
            except socket.timeout:
                latency_ms = (time.time() - t0) * 1000
                status = f"{RED}{BOLD}TIMEOUT (ISOLATED){RESET}"
                blocked_count += 1
            except Exception as e:
                latency_ms = (time.time() - t0) * 1000
                status = f"{RED}ERROR ({e.__class__.__name__}){RESET}"
                blocked_count += 1
            finally:
                s.close()
                
            print(f"{probe_count:<5} | {port:<8} | {status:<25} | {latency_ms:6.1f}ms        | {target_ip}")
            
            if delay_s > 0:
                time.sleep(delay_s)
                
    except KeyboardInterrupt:
        print(f"\n{YELLOW}[!] Scan interrupted by user.{RESET}")

    print("\n" + "="*75)
    print(f"{BOLD}SCAN SUMMARY:{RESET}")
    print(f"  Total Probes Sent   : {probe_count}")
    print(f"  Ports Reached       : {open_count}")
    print(f"  Probes Dropped/Blocked: {blocked_count}")
    if blocked_count > 0:
        print(f"  {RED}{BOLD}>>> TARGET ACTIVATED DEFENSE: TRAFFIC WAS INTERCEPTED & QUARANTINED <<<{RESET}")
    print("="*75 + "\n")

def main():
    print_banner()
    parser = argparse.ArgumentParser(description="Port Scan Attack Simulator for Quantum NIDS Demo")
    parser.add_argument("--target", type=str, default="127.0.0.1", help="Target IP address (default: 127.0.0.1)")
    parser.add_argument("--start", type=int, default=8000, help="Starting port (default: 8000)")
    parser.add_argument("--end", type=int, default=8015, help="Ending port (default: 8015)")
    parser.add_argument("--delay", type=float, default=0.08, help="Delay between probes in seconds (default: 0.08s)")
    parser.add_argument("--auto", action="store_true", help="Run without prompt")
    
    args = parser.parse_args()

    target_ip = args.target
    start_p = args.start
    end_p = args.end
    delay = args.delay

    if not args.auto:
        print(f"Configure attack scenario (press ENTER to use defaults):")
        inp_ip = input(f"Enter Target IP [{target_ip}]: ").strip()
        if inp_ip:
            target_ip = inp_ip
            
        inp_start = input(f"Enter Start Port [{start_p}]: ").strip()
        if inp_start.isdigit():
            start_p = int(inp_start)
            
        inp_end = input(f"Enter End Port [{end_p}]: ").strip()
        if inp_end.isdigit():
            end_p = int(inp_end)
            
        inp_speed = input(f"Select Scan Speed: [1] Fast Sweep (Default), [2] Aggressive Flood, [3] Stealth: ").strip()
        if inp_speed == "2":
            delay = 0.01
        elif inp_speed == "3":
            delay = 0.3
        else:
            delay = 0.08

    print(f"\n{RED}[!] INITIATING MALICIOUS PORT SCAN RECONNAISSANCE PROBE...{RESET}")
    time.sleep(0.5)
    scan_target(target_ip, start_p, end_p, delay)

if __name__ == "__main__":
    main()
