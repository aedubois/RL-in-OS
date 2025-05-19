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
        """Initialize the GUI."""
        self.root = root
        self.root.title("Kernel Tune Interface")
        self.root.geometry("500x900")
        self.processes = []

        # Metrics and logs collection
        self.t0 = time.time()
        self.metrics = []
        self.logs = []
        self.agent_logs = []
        self.collecting = True

        # Initialize the EventAgent
        self.agent = EventAgent()

        # Header
        header = ttk.Label(root, text="Kernel Tune Interface", font=("Arial", 16, "bold"))
        header.pack(pady=10)

        # Button to open the monitoring window
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

        # Buttons for each action
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

        # Button to stop all stress and clean resources
        ttk.Button(root, text="Stop All Stress", command=self.clean_resources).pack(pady=10)

        # Exit button
        ttk.Button(root, text="Exit", command=self.exit_application).pack(pady=10)

        # Start metrics collection
        self.collect_metrics()

    def update_timer(self):
        """Update the elapsed time display."""
        elapsed = int(time.time() - self.t0)
        self.time_label.config(text=f"Elapsed time: {elapsed} s")
        self.root.after(1000, self.update_timer)

    def reset_timer(self):
        """Reset the timer and clear logs."""
        self.t0 = time.time()
        self.metrics = []
        self.logs = []
        self.agent_logs = []
        messagebox.showinfo("Timer", "Timer reset.")

    def collect_metrics(self):
        """Collect system metrics and store them in the metrics list."""
        if self.collecting:
            t = time.time() - self.t0
            cpu = psutil.cpu_percent()
            ram = psutil.virtual_memory().percent
            swap = psutil.swap_memory().percent
            temp = None
            try:
                temps = psutil.sensors_temperatures()
                if temps:
                    for name, entries in temps.items():
                        for entry in entries:
                            if hasattr(entry, 'label') and ('cpu' in entry.label.lower() or 'core' in entry.label.lower()):
                                temp = entry.current
                                break
                        if temp is not None:
                            break
                    if temp is None:
                        for entries in temps.values():
                            if entries:
                                temp = entries[0].current
                                break
            except Exception:
                temp = None
            disk = psutil.disk_usage('/').percent
            net = psutil.net_io_counters()
            net_sent = net.bytes_sent
            net_recv = net.bytes_recv
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
        """Log the user action with the current time."""
        t = time.time() - self.t0
        self.logs.append({"time": t, "action": action})

    def log_agent_reaction(self, reaction):
        """Log the agent's reaction with the current time."""
        t = time.time() - self.t0
        self.agent_logs.append({"time": t, "reaction": reaction})

    def generate_plot(self):
        """Generate a plot of the collected metrics."""
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

        # User action markers 
        for log in self.logs:
            plt.scatter(log["time"], 1.02, marker="v", color="black", label="User Action" if log == self.logs[0] else "")
            plt.text(log["time"], 1.05, log["action"], rotation=90, verticalalignment='bottom', fontsize=7)

        # Agent reaction markers 
        for agent_log in self.agent_logs:
            plt.scatter(agent_log["time"], -0.08, marker="*", color="purple", s=120, label="Agent Reaction" if agent_log == self.agent_logs[0] else "")
            plt.text(agent_log["time"], -0.13, agent_log["reaction"], rotation=90, verticalalignment='top', fontsize=7, color="purple")

        plt.xlabel("Time (s)")
        plt.ylabel("Normalized metrics")
        plt.ylim(-0.18, 1.15)
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
        """Save the collected metrics to a CSV file."""
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
        """Save the action logs to a text file."""
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

    def simulate_cpu_stress(self):
        """Simulate CPU stress by creating multiple processes."""
        def _cpu_stress_worker():
            while True:
                pass
        for _ in range(multiprocessing.cpu_count() // 2):
            process = multiprocessing.Process(target=_cpu_stress_worker)
            process.start()
            self.processes.append(process)
        self.log_action("Simulate CPU Stress")
        time.sleep(1)
        reaction = self.agent.handle_event("Simulate CPU Stress", plot=True)
        self.log_agent_reaction(reaction)
        messagebox.showinfo("Action", "CPU stress simulation started.")

    def simulate_memory_stress(self):
        """Simulate memory stress by allocating large amounts of memory."""
        self.memory_stress = [" " * 10**6 for _ in range(10**4)]
        self.log_action("Simulate Memory Stress")
        time.sleep(1)
        reaction = self.agent.handle_event("Simulate Memory Stress", plot=True)
        self.log_agent_reaction(reaction)
        messagebox.showinfo("Action", "Memory stress simulation started.")

    def simulate_disk_io_stress(self):
        """Simulate disk I/O stress by writing large files."""
        subprocess.run(["dd", "if=/dev/zero", "of=/tmp/largefile", "bs=1M", "count=256"])
        self.log_action("Simulate Disk I/O Stress")
        time.sleep(1)
        reaction = self.agent.handle_event("Simulate Disk I/O Stress", plot=True)
        self.log_agent_reaction(reaction)
        messagebox.showinfo("Action", "Disk I/O stress simulation completed.")

    def simulate_gpu_load(self):
        """Simulate GPU load using glxgears."""
        try:
            subprocess.Popen(["glxgears"], stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
            self.log_action("Simulate GPU Load")
            time.sleep(1)
            reaction = self.agent.handle_event("Simulate GPU Load", plot=True)
            self.log_agent_reaction(reaction)
            messagebox.showinfo("Action", "GPU load simulation started.")
        except FileNotFoundError:
            messagebox.showerror("Error", "glxgears not found. Please install it.")

    def run_compilation_task(self):
        """Run a compilation task using g++."""
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
            self.log_action("Run Compilation Task")
            time.sleep(1)
            reaction = self.agent.handle_event("Run Compilation Task", plot=True)
            self.log_agent_reaction(reaction)

    def simulate_network_flood(self):
        """Simulate network flood using ping."""
        try:
            subprocess.Popen(["ping", "-f", "8.8.8.8"], stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
            self.log_action("Simulate Network Flood")
            time.sleep(1)
            reaction = self.agent.handle_event("Simulate Network Flood", plot=True)
            self.log_agent_reaction(reaction)
            messagebox.showinfo("Action", "Network flood simulation started.")
        except FileNotFoundError:
            messagebox.showerror("Error", "Ping command not found.")

    def fill_disk_until_threshold(self):
        """Fill the disk until a certain threshold."""
        try:
            subprocess.run(["dd", "if=/dev/zero", "of=/tmp/fillfile", "bs=1M", "count=1024"])
            self.log_action("Fill Disk Until Threshold")
            time.sleep(1)
            reaction = self.agent.handle_event("Fill Disk Until Threshold", plot=True)
            self.log_agent_reaction(reaction)
            messagebox.showinfo("Action", "Disk filling simulation completed.")
        except FileNotFoundError:
            messagebox.showerror("Error", "dd command not found.")

    def play_streaming_video(self):
        """Play a streaming video using VLC."""
        try:
            subprocess.Popen(["vlc", "https://www.youtube.com/watch?v=dQw4w9WgXcQ"], stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
            self.log_action("Play Streaming Video")
            time.sleep(1)
            reaction = self.agent.handle_event("Play Streaming Video", plot=True)
            self.log_agent_reaction(reaction)
            messagebox.showinfo("Action", "Streaming video started.")
        except FileNotFoundError:
            messagebox.showerror("Error", "VLC not found. Please install it.")

    def spawn_multiple_processes(self):
        """Spawn multiple processes to simulate high load."""
        try:
            for _ in range(100):
                process = subprocess.Popen(["sleep", "100"])
                self.processes.append(process)
            self.log_action("Spawn Multiple Processes")
            time.sleep(1)
            reaction = self.agent.handle_event("Spawn Multiple Processes", plot=True)
            self.log_agent_reaction(reaction)
            messagebox.showinfo("Action", "Spawned multiple processes.")
        except Exception as e:
            messagebox.showerror("Error", f"An unexpected error occurred:\n{e}")

    def simulate_disk_latency(self):
        """Simulate disk latency using stress-ng."""
        try:
            subprocess.Popen(["stress-ng", "--io", "1", "--timeout", "30"], stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
            self.log_action("Simulate Disk Latency")
            time.sleep(1)
            reaction = self.agent.handle_event("Simulate Disk Latency", plot=True)
            self.log_agent_reaction(reaction)
            messagebox.showinfo("Action", "Disk latency simulation started.")
        except FileNotFoundError:
            messagebox.showerror("Error", "stress-ng not found. Please install it.")

    def stress_tmpfs(self):
        """Stress the tmpfs filesystem."""
        try:
            subprocess.run(["dd", "if=/dev/zero", "of=/dev/shm/tmpfile", "bs=1M", "count=512"])
            self.log_action("Stress Tmpfs")
            time.sleep(1)
            reaction = self.agent.handle_event("Stress Tmpfs", plot=True)
            self.log_agent_reaction(reaction)
            messagebox.showinfo("Action", "Tmpfs stress simulation completed.")
        except FileNotFoundError:
            messagebox.showerror("Error", "dd command not found.")

    def clean_resources(self):
        """Stop all stress processes and clean resources."""
        self.agent.clean_resources()
        reaction = self.agent.handle_event("Stop All Stress", plot=True)
        self.log_agent_reaction(reaction)
        self.log_action("Stop All Stress")
        messagebox.showinfo("Action", "Resources cleaned.")

    def open_monitor(self):
        """Open the system monitor window."""
        monitor_root = tk.Toplevel(self.root)
        SystemMonitorGUI(monitor_root)

    def exit_application(self):
        """Exit the application."""
        self.clean_resources()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = KernelTuneGUI(root)
    root.mainloop()
