import tkinter as tk
from tkinter import ttk
import psutil
import threading
import time
import subprocess
import os

class SystemMonitorGUI:
    def __init__(self, root):
        """Initializes the GUI for the system monitor."""
        self.root = root
        self.root.title("System Monitor")
        self.root.geometry("500x600")

        # Create labels for metrics
        self.cpu_label = ttk.Label(root, text="CPU Usage: ", font=("Arial", 12))
        self.cpu_label.pack(pady=5)

        self.cpu_freq_label = ttk.Label(root, text="CPU Frequency: ", font=("Arial", 12))
        self.cpu_freq_label.pack(pady=5)

        self.temp_label = ttk.Label(root, text="CPU Temperature: ", font=("Arial", 12))
        self.temp_label.pack(pady=5)

        self.disk_label = ttk.Label(root, text="Disk Usage: ", font=("Arial", 12))
        self.disk_label.pack(pady=5)

        self.page_faults_label = ttk.Label(root, text="Page Faults/s: ", font=("Arial", 12))
        self.page_faults_label.pack(pady=5)

        self.interrupts_label = ttk.Label(root, text="Interrupts/s: ", font=("Arial", 12))
        self.interrupts_label.pack(pady=5)

        self.network_label = ttk.Label(root, text="Network Throughput: ", font=("Arial", 12))
        self.network_label.pack(pady=5)

        self.io_queue_label = ttk.Label(root, text="I/O Queue Length: ", font=("Arial", 12))
        self.io_queue_label.pack(pady=5)

        self.memory_label = ttk.Label(root, text="Memory Usage: ", font=("Arial", 12))
        self.memory_label.pack(pady=5)

        self.swap_label = ttk.Label(root, text="Swap Usage: ", font=("Arial", 12))
        self.swap_label.pack(pady=5)

        self.load_label = ttk.Label(root, text="Load Average: ", font=("Arial", 12))
        self.load_label.pack(pady=5)

        self.io_wait_label = ttk.Label(root, text="I/O Wait: ", font=("Arial", 12))
        self.io_wait_label.pack(pady=5)

        self.process_label = ttk.Label(root, text="Active Processes: ", font=("Arial", 12))
        self.process_label.pack(pady=5)

        # Start a thread to update metrics
        self.running = True
        self.update_thread = threading.Thread(target=self.update_metrics, daemon=True)
        self.update_thread.start()

        # Handle window close event
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def update_metrics(self):
        """Updates the system metrics in the GUI."""
        while self.running:
            try:
                # Collect metrics
                cpu_usage = psutil.cpu_percent(interval=1)
                cpu_freq = psutil.cpu_freq().current if psutil.cpu_freq() else "N/A"

                # CPU temperature
                temp = "N/A"
                temperatures = psutil.sensors_temperatures()
                if "coretemp" in temperatures:
                    core_temps = temperatures["coretemp"]
                    if core_temps:
                        temp = core_temps[0].current

                # Disk usage
                disk_usage = psutil.disk_usage('/').percent

                # Page faults and interrupts
                vmstat_output = subprocess.check_output(["vmstat", "1", "2"]).decode().splitlines()
                if len(vmstat_output) > 2:
                    stats = vmstat_output[-1].split()
                    page_faults = stats[9]
                    interrupts = stats[11]
                else:
                    page_faults = "N/A"
                    interrupts = "N/A"

                # Network throughput
                net_io = psutil.net_io_counters()
                network_throughput = f"Sent: {net_io.bytes_sent / 1024:.2f} KB, Recv: {net_io.bytes_recv / 1024:.2f} KB"

                # I/O queue length
                io_queue = subprocess.check_output(["iostat", "-x", "1", "2"]).decode().splitlines()
                if len(io_queue) > 3:
                    io_stats = io_queue[-1].split()
                    io_queue_length = io_stats[8] if len(io_stats) > 8 else "N/A"
                else:
                    io_queue_length = "N/A"

                memory_usage = psutil.virtual_memory().percent
                self.memory_label.config(text=f"Memory Usage: {memory_usage:.2f}%")

                swap_usage = psutil.swap_memory().percent
                self.swap_label.config(text=f"Swap Usage: {swap_usage:.2f}%")

                load_avg = os.getloadavg()[0]  # Load average sur 1 minute
                self.load_label.config(text=f"Load Average: {load_avg:.2f}")

                cpu_times = psutil.cpu_times()
                io_wait = cpu_times.iowait if hasattr(cpu_times, 'iowait') else "N/A"
                self.io_wait_label.config(text=f"I/O Wait: {io_wait:.2f} sec")

                process_count = len(psutil.pids())
                self.process_label.config(text=f"Active Processes: {process_count}")

                # Update labels
                self.cpu_label.config(text=f"CPU Usage: {cpu_usage:.2f}%")
                self.cpu_freq_label.config(text=f"CPU Frequency: {cpu_freq:.2f} MHz")
                self.temp_label.config(text=f"CPU Temperature: {temp}°C")
                self.disk_label.config(text=f"Disk Usage: {disk_usage:.2f}%")
                self.page_faults_label.config(text=f"Page Faults/s: {page_faults}")
                self.interrupts_label.config(text=f"Interrupts/s: {interrupts}")
                self.network_label.config(text=f"Network Throughput: {network_throughput}")
                self.io_queue_label.config(text=f"I/O Queue Length: {io_queue_length}")

                time.sleep(1)
            except Exception as e:
                print(f"Error updating metrics: {e}")

    def close(self):
        """Stops the update thread and closes the window."""
        self.running = False
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = SystemMonitorGUI(root)
    root.mainloop()
