import tkinter as tk
from tkinter import ttk, messagebox
import subprocess
import os
import multiprocessing
import psutil
import time
import matplotlib.pyplot as plt
import numpy as np
import csv
import shutil
from monitor_interface import SystemMonitorGUI
from agent import EventAgent

class KernelTuneGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Kernel Tune Interface")
        self.root.geometry("500x900")  # Enlarged window
        self.processes = []

        # --- For metrics and logs collection ---
        self.t0 = time.time()
        self.metrics = []
        self.logs = []
        self.collecting = True

        # Initialize the EventAgent
        self.agent = EventAgent()

        # Header
        header = ttk.Label(root, text="Kernel Tune Interface", font=("Arial", 16, "bold"))
        header.pack(pady=10)

        # Button to open the monitoring window (at the top)
        ttk.Button(root, text="Open System Monitor", command=self.open_monitor).pack(pady=10)

        # Dynamic elapsed time display
        self.time_label = ttk.Label(root, text="Elapsed time: 0 s", font=("Arial", 12))
        self.time_label.pack(pady=5)
        self.update_timer()

        # Buttons for logs/plots/reset
        ttk.Button(root, text="Generate plot", command=self.generate_plot).pack(pady=5)
        ttk.Button(root, text="Save actions logs", command=self.save_actions_logs).pack(pady=5)
        ttk.Button(root, text="Save metrics logs", command=self.save_metrics_csv).pack(pady=5)
        ttk.Button(root, text="Reset timer", command=self.reset_timer).pack(pady=5)

        # Create buttons for each action
        actions_frame = ttk.LabelFrame(root, text="Actions", padding=(10, 10))
        actions_frame.pack(fill="both", expand=True, padx=10, pady=10)

        ttk.Button(actions_frame, text="Simulate CPU Stress", command=self.simulate_cpu_stress).pack(pady=5)
        ttk.Button(actions_frame, text="Simulate Memory Stress", command=self.simulate_memory_stress).pack(pady=5)
        ttk.Button(actions_frame, text="Simulate Disk I/O Stress", command=self.simulate_disk_io_stress).pack(pady=5)
        ttk.Button(actions_frame, text="Simulate GPU Load", command=self.simulate_gpu_load).pack(pady=5)
        ttk.Button(actions_frame, text="Run Compilation Task", command=self.run_compilation_task).pack(pady=5)
        ttk.Button(actions_frame, text="Simulate Network Flood", command=self.simulate_network_flood).pack(pady=5)
        ttk.Button(actions_frame, text="Fill Disk Until Threshold", command=self.fill_disk_until_threshold).pack(pady=5)
        ttk.Button(actions_frame, text="Play Streaming Video", command=self.play_streaming_video).pack(pady=5)
        ttk.Button(actions_frame, text="Spawn Multiple Processes", command=self.spawn_multiple_processes).pack(pady=5)
        ttk.Button(actions_frame, text="Simulate Disk Latency", command=self.simulate_disk_latency).pack(pady=5)
        ttk.Button(actions_frame, text="Stress Tmpfs", command=self.stress_tmpfs).pack(pady=5)

        # Button to stop all stress (at the bottom)
        ttk.Button(root, text="Stop All Stress", command=self.clean_resources).pack(pady=10)

        # Exit button
        ttk.Button(root, text="Exit", command=self.exit_application).pack(pady=10)

        # Start metrics collection
        self.collect_metrics()

    # --- Metrics and logs ---
    def update_timer(self):
        elapsed = int(time.time() - self.t0)
        self.time_label.config(text=f"Elapsed time: {elapsed} s")
        self.root.after(1000, self.update_timer)

    def reset_timer(self):
        self.t0 = time.time()
        self.metrics = []
        self.logs = []
        messagebox.showinfo("Timer", "Timer reset.")

    def collect_metrics(self):
        if self.collecting:
            t = time.time() - self.t0
            cpu = psutil.cpu_percent()
            ram = psutil.virtual_memory().percent
            swap = psutil.swap_memory().percent
            # Try to get CPU temperature (Linux only, may require sensors)
            temp = None
            try:
                temps = psutil.sensors_temperatures()
                if temps:
                    # Try to find a CPU sensor
                    for name, entries in temps.items():
                        for entry in entries:
                            if hasattr(entry, 'label') and ('cpu' in entry.label.lower() or 'core' in entry.label.lower()):
                                temp = entry.current
                                break
                        if temp is not None:
                            break
                    if temp is None:
                        # Fallback: take first available temperature
                        for entries in temps.values():
                            if entries:
                                temp = entries[0].current
                                break
            except Exception:
                temp = None
            # Additional metrics
            disk = psutil.disk_usage('/').percent
            net = psutil.net_io_counters()
            net_sent = net.bytes_sent
            net_recv = net.bytes_recv
            # Free space in GB
            free_disk_gb = shutil.disk_usage('/').free / (1024**3)
            used_disk_gb = shutil.disk_usage('/').used / (1024**3)
            total_disk_gb = shutil.disk_usage('/').total / (1024**3)
            load1, load5, load15 = os.getloadavg()
            procs = len(psutil.pids())
            ctx_switches = psutil.cpu_stats().ctx_switches
            interrupts = psutil.cpu_stats().interrupts
            soft_interrupts = psutil.cpu_stats().soft_interrupts
            self.metrics.append({
                "time": t,
                "cpu": cpu,
                "ram": ram,
                "swap": swap,
                "temp": temp,
                "disk": disk,
                "net_sent": net_sent,
                "net_recv": net_recv,
                "free_disk_gb": free_disk_gb,
                "used_disk_gb": used_disk_gb,
                "total_disk_gb": total_disk_gb,
                "load1": load1,
                "load5": load5,
                "load15": load15,
                "procs": procs,
                "ctx_switches": ctx_switches,
                "interrupts": interrupts,
                "soft_interrupts": soft_interrupts
            })
            self.root.after(1000, self.collect_metrics)

    def log_action(self, action):
        t = time.time() - self.t0
        self.logs.append({"time": t, "action": action})

    def generate_plot(self):
        if not self.metrics:
            messagebox.showerror("Plot", "No data to display.")
            return
        times = [m["time"] for m in self.metrics]
        cpu = np.array([m["cpu"] for m in self.metrics])
        ram = np.array([m["ram"] for m in self.metrics])
        disk = np.array([m["disk"] for m in self.metrics])
        temp = np.array([m["temp"] if m["temp"] is not None else np.nan for m in self.metrics])

        # Normalization
        cpu_norm = (cpu - cpu.min()) / (cpu.max() - cpu.min() + 1e-6)
        ram_norm = (ram - ram.min()) / (ram.max() - ram.min() + 1e-6)
        disk_norm = (disk - disk.min()) / (disk.max() - disk.min() + 1e-6)
        if not np.isnan(temp).all():
            temp_norm = (temp - np.nanmin(temp)) / (np.nanmax(temp) - np.nanmin(temp) + 1e-6)
        else:
            temp_norm = None

        plt.figure(figsize=(12,6))
        plt.plot(times, cpu_norm, label="CPU %", color='tab:blue')
        plt.plot(times, ram_norm, label="RAM %", color='tab:orange')
        plt.plot(times, disk_norm, label="Disk Usage %", color='tab:green')
        if temp_norm is not None:
            plt.plot(times, temp_norm, label="CPU Temp", color='tab:red')
        plt.grid(alpha=0.2)

        # Discrete action markers (triangle)
        for log in self.logs:
            plt.scatter(log["time"], 1.02, marker="v", color="black")
            plt.text(log["time"], 1.05, log["action"], rotation=90, verticalalignment='bottom', fontsize=7)

        plt.xlabel("Time (s)")
        plt.ylabel("Normalized metrics")
        plt.ylim(0, 1.15)
        plt.legend()
        plt.tight_layout()
        plots_dir = os.path.join(os.path.dirname(__file__), "plots")
        os.makedirs(plots_dir, exist_ok=True)
        i = 1
        while os.path.exists(os.path.join(plots_dir, f"plot_{i}.png")):
            i += 1
        plot_path = os.path.join(plots_dir, f"plot_{i}.png")
        plt.savefig(plot_path)
        plt.show()
        messagebox.showinfo("Plot", f"Plot saved in {plot_path}")

    def save_metrics_csv(self):
        logs_dir = os.path.join(os.path.dirname(__file__), "metrics_logs")
        os.makedirs(logs_dir, exist_ok=True)
        i = 1
        while os.path.exists(os.path.join(logs_dir, f"metrics_{i}.csv")):
            i += 1
        csv_path = os.path.join(logs_dir, f"metrics_{i}.csv")
        with open(csv_path, "w", newline="") as csvfile:
            fieldnames = [
                "time", "cpu", "ram", "swap", "temp", "disk", "net_sent", "net_recv",
                "free_disk_gb", "used_disk_gb", "total_disk_gb",
                "load1", "load5", "load15", "procs", "ctx_switches", "interrupts", "soft_interrupts"
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for m in self.metrics:
                writer.writerow(m)
        messagebox.showinfo("Metrics", f"Metrics saved in {csv_path}")

    def save_actions_logs(self):
        logs_dir = os.path.join(os.path.dirname(__file__), "actions_logs")
        os.makedirs(logs_dir, exist_ok=True)
        i = 1
        while os.path.exists(os.path.join(logs_dir, f"actions_{i}.txt")):
            i += 1
        log_path = os.path.join(logs_dir, f"actions_{i}.txt")
        with open(log_path, "w") as f:
            for log in self.logs:
                f.write(f"{log['time']:.2f}s : {log['action']}\n")
        messagebox.showinfo("Logs", f"Action logs saved in {log_path}")

    # --- Actions (ajoute log_action à chaque action) ---
    def simulate_cpu_stress(self):
        def _cpu_stress_worker():
            while True:
                pass
        print("Simulating CPU stress...")
        for _ in range(multiprocessing.cpu_count() // 2):
            process = multiprocessing.Process(target=_cpu_stress_worker)
            process.start()
            self.processes.append(process)
        self.agent.handle_event("Simulate CPU Stress")
        self.log_action("Simulate CPU Stress")
        messagebox.showinfo("Action", "CPU stress simulation started.")

    def simulate_memory_stress(self):
        print("Simulating memory stress...")
        self.memory_stress = [" " * 10**6 for _ in range(10**4)]
        self.agent.handle_event("Simulate Memory Stress")
        self.log_action("Simulate Memory Stress")
        messagebox.showinfo("Action", "Memory stress simulation started.")

    def simulate_disk_io_stress(self):
        print("Simulating disk I/O stress...")
        subprocess.run(["dd", "if=/dev/zero", "of=/tmp/largefile", "bs=1M", "count=256"])
        self.agent.handle_event("Simulate Disk I/O Stress")
        self.log_action("Simulate Disk I/O Stress")
        messagebox.showinfo("Action", "Disk I/O stress simulation completed.")

    def simulate_gpu_load(self):
        print("Simulating GPU load...")
        try:
            subprocess.Popen(["glxgears"], stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
            self.agent.handle_event("Simulate GPU Load")
            self.log_action("Simulate GPU Load")
            messagebox.showinfo("Action", "GPU load simulation started.")
        except FileNotFoundError:
            messagebox.showerror("Error", "glxgears not found. Please install it.")

    def run_compilation_task(self):
        print("Running compilation task...")
        utils_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "utils")
        code_file = os.path.join(utils_dir, "code.cpp")
        if not os.path.exists(code_file):
            messagebox.showerror("Error", f"code.cpp introuvable dans {utils_dir}")
            return
        try:
            result = subprocess.run(["g++", "-o", "code", code_file], cwd=utils_dir, capture_output=True, text=True)
            if result.returncode == 0:
                messagebox.showinfo("Action", "Compilation task completed successfully.")
            else:
                messagebox.showerror("Error", f"Compilation failed:\n{result.stderr}")
        except FileNotFoundError:
            messagebox.showerror("Error", "g++ not found. Please install it.")
        except Exception as e:
            messagebox.showerror("Error", f"An unexpected error occurred:\n{e}")
        finally:
            self.agent.handle_event("Run Compilation Task")
            self.log_action("Run Compilation Task")

    def simulate_network_flood(self):
        print("Simulating network flood...")
        try:
            subprocess.Popen(["ping", "-f", "8.8.8.8"], stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
            self.agent.handle_event("Simulate Network Flood")
            self.log_action("Simulate Network Flood")
            messagebox.showinfo("Action", "Network flood simulation started.")
        except FileNotFoundError:
            messagebox.showerror("Error", "Ping command not found.")

    def fill_disk_until_threshold(self):
        print("Filling disk until threshold...")
        try:
            subprocess.run(["dd", "if=/dev/zero", "of=/tmp/fillfile", "bs=1M", "count=1024"])
            self.agent.handle_event("Fill Disk Until Threshold")
            self.log_action("Fill Disk Until Threshold")
            messagebox.showinfo("Action", "Disk filling simulation completed.")
        except FileNotFoundError:
            messagebox.showerror("Error", "dd command not found.")

    def play_streaming_video(self):
        print("Playing streaming video...")
        try:
            subprocess.Popen(["vlc", "https://www.youtube.com/watch?v=dQw4w9WgXcQ"], stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
            self.agent.handle_event("Play Streaming Video")
            self.log_action("Play Streaming Video")
            messagebox.showinfo("Action", "Streaming video started.")
        except FileNotFoundError:
            messagebox.showerror("Error", "VLC not found. Please install it.")

    def spawn_multiple_processes(self):
        print("Spawning multiple processes...")
        try:
            for _ in range(100):
                process = subprocess.Popen(["sleep", "100"])
                self.processes.append(process)
            self.agent.handle_event("Spawn Multiple Processes")
            self.log_action("Spawn Multiple Processes")
            messagebox.showinfo("Action", "Spawned multiple processes.")
        except Exception as e:
            messagebox.showerror("Error", f"An unexpected error occurred:\n{e}")

    def simulate_disk_latency(self):
        print("Simulating disk latency...")
        try:
            subprocess.Popen(["stress-ng", "--io", "1", "--timeout", "30"], stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
            self.agent.handle_event("Simulate Disk Latency")
            self.log_action("Simulate Disk Latency")
            messagebox.showinfo("Action", "Disk latency simulation started.")
        except FileNotFoundError:
            messagebox.showerror("Error", "stress-ng not found. Please install it.")

    def stress_tmpfs(self):
        print("Stressing tmpfs...")
        try:
            subprocess.run(["dd", "if=/dev/zero", "of=/dev/shm/tmpfile", "bs=1M", "count=512"])
            self.agent.handle_event("Stress Tmpfs")
            self.log_action("Stress Tmpfs")
            messagebox.showinfo("Action", "Tmpfs stress simulation completed.")
        except FileNotFoundError:
            messagebox.showerror("Error", "dd command not found.")

    def clean_resources(self):
        print("Cleaning up resources...")
        for process in self.processes:
            if isinstance(process, multiprocessing.Process):
                process.terminate()
        self.processes = []
        if os.path.exists("/tmp/largefile"):
            os.remove("/tmp/largefile")
        if os.path.exists("/tmp/fillfile"):
            os.remove("/tmp/fillfile")
        if hasattr(self, "memory_stress"):
            del self.memory_stress
        os.system("pkill stress")
        os.system("pkill yes")
        os.system("pkill glxgears")
        os.system("pkill vlc")
        os.system("pkill ping")
        self.agent.handle_event("Stop All Stress")
        self.log_action("Stop All Stress")
        messagebox.showinfo("Action", "Resources cleaned.")

    def open_monitor(self):
        monitor_root = tk.Toplevel(self.root)
        SystemMonitorGUI(monitor_root)

    def exit_application(self):
        self.clean_resources()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = KernelTuneGUI(root)
    root.mainloop()
