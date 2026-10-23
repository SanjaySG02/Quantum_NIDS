#!/usr/bin/env python3
"""
Quantum NIDS - Threat Containment HUD Window
Spawns as an independent top-level process so it never conflicts with
the detector's network loop and can be triggered multiple times seamlessly.
"""

import sys
import json
import subprocess
import tkinter as tk

def lock_workstation():
    try:
        subprocess.run(["rundll32.exe", "user32.dll,LockWorkStation"], check=False)
    except Exception as e:
        print(f"Error locking workstation: {e}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python alert_hud.py '<json_threat_data>' [--auto-lock]")
        sys.exit(1)

    try:
        threat_data = json.loads(sys.argv[1])
    except Exception:
        threat_data = {}

    auto_lock = "--auto-lock" in sys.argv
    countdown_sec = int(threat_data.get("countdown", 7))

    root = tk.Tk()
    root.title("QUANTUM NIDS - AUTONOMOUS CONTAINMENT ACTIVE")
    root.geometry("640x530")
    root.configure(bg="#090d16")
    root.resizable(False, False)

    # Force window to top and bring to front
    root.lift()
    root.attributes('-topmost', True)
    root.focus_force()

    # Main container frame
    frame = tk.Frame(root, bg="#0d1527", bd=2, relief="solid", highlightbackground="#ef4444", highlightthickness=3)
    frame.pack(fill="both", expand=True, padx=15, pady=15)

    # Header Title
    title_lbl = tk.Label(
        frame,
        text="⚠️ QUANTUM INTRUSION CONTAINMENT ACTIVATED ⚠️",
        font=("Segoe UI", 13, "bold"),
        fg="#ef4444",
        bg="#0d1527"
    )
    title_lbl.pack(pady=(15, 5))

    sub_lbl = tk.Label(
        frame,
        text=f"Autonomous Response: {threat_data.get('threat_title', 'Malicious Reconnaissance Port Sweep Intercepted')}",
        font=("Segoe UI", 9),
        fg="#94a3b8",
        bg="#0d1527"
    )
    sub_lbl.pack(pady=(0, 15))

    # Metrics Table Frame
    grid_frame = tk.Frame(frame, bg="#131e36", bd=1, relief="ridge")
    grid_frame.pack(fill="x", padx=20, pady=5)

    f_att = threat_data.get('f_att', 0.0)
    sim_score = threat_data.get('sim_score', 0.0)
    thresh = threat_data.get('threshold', -0.28)
    conf = threat_data.get('confidence', 0.0)
    latency = threat_data.get('latency_ms', 0.0)

    row_data = [
        ("Threat Classification:", f"{threat_data.get('classification', 'MALICIOUS INTRUSION (PORT SCAN)')}", "#ef4444"),
        ("Attacker Source IP:", f"{threat_data.get('attacker_ip', '127.0.0.1')}", "#f87171"),
        ("Target Ports Probed:", f"{threat_data.get('ports_scanned', 'Multiple')}", "#f8fafc"),
        ("Quantum Attack Overlap (F_att):", f"{f_att:.4f} ({f_att*100:.1f}%)", "#38bdf8"),
        ("Quantum Similarity Score:", f"{sim_score:+.4f} (Threshold: {thresh:.2f})", "#4ade80"),
        ("Decision Confidence:", f"{conf:.1f}%", "#facc15"),
        ("Quantum Circuit Latency:", f"{latency:.1f} ms", "#cbd5e1"),
        ("Firewall Quarantine Status:", "ACTIVE (Inbound Rule Applied)", "#4ade80"),
    ]

    for i, (label_txt, val_txt, val_color) in enumerate(row_data):
        lbl = tk.Label(grid_frame, text=label_txt, font=("Segoe UI", 9, "bold"), fg="#94a3b8", bg="#131e36", anchor="w")
        lbl.grid(row=i, column=0, padx=10, pady=3, sticky="w")
        val = tk.Label(grid_frame, text=val_txt, font=("Segoe UI", 9, "bold"), fg=val_color, bg="#131e36", anchor="w")
        val.grid(row=i, column=1, padx=10, pady=3, sticky="w")

    # Countdown label
    timer_var = tk.StringVar()
    if auto_lock:
        timer_var.set(f"Locking Workstation in {countdown_sec} seconds...")
    else:
        timer_var.set("[SAFE TESTING MODE: Auto-Lock Disabled]")

    countdown_lbl = tk.Label(frame, textvariable=timer_var, font=("Segoe UI", 11, "bold"), fg="#facc15", bg="#0d1527")
    countdown_lbl.pack(pady=12)

    remaining = [countdown_sec]
    locked = [False]

    def tick():
        if locked[0]:
            return
        if remaining[0] > 0:
            timer_var.set(f"Automated Workstation Lock in {remaining[0]}s...")
            remaining[0] -= 1
            root.after(1000, tick)
        else:
            timer_var.set("Workstation Locked.")
            locked[0] = True
            lock_workstation()

    if auto_lock:
        root.after(1000, tick)

    # Buttons
    btn_frame = tk.Frame(frame, bg="#0d1527")
    btn_frame.pack(pady=(5, 10))

    def manual_lock():
        locked[0] = True
        timer_var.set("Workstation Locked.")
        lock_workstation()

    def dismiss():
        locked[0] = True
        root.destroy()

    lock_btn = tk.Button(
        btn_frame,
        text="🔒 LOCK WORKSTATION NOW",
        font=("Segoe UI", 10, "bold"),
        bg="#ef4444",
        fg="#ffffff",
        activebackground="#dc2626",
        activeforeground="#ffffff",
        padx=15,
        pady=6,
        command=manual_lock
    )
    lock_btn.pack(side="left", padx=10)

    dismiss_btn = tk.Button(
        btn_frame,
        text="Dismiss / Reset",
        font=("Segoe UI", 9),
        bg="#334155",
        fg="#f8fafc",
        activebackground="#475569",
        padx=12,
        pady=6,
        command=dismiss
    )
    dismiss_btn.pack(side="left", padx=10)

    root.mainloop()

if __name__ == "__main__":
    main()
