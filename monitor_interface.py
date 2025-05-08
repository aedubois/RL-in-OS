import tkinter as tk
from tkinter import ttk
import psutil
import threading
import time

class SystemMonitorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("System Monitor")
        self.root.geometry("400x300")

        # Create labels for metrics
        self.cpu_label = ttk.Label(root, text="CPU Usage: ", font=("Arial", 12))
        self.cpu_label.pack(pady=10)

        self.memory_label = ttk.Label(root, text="Memory Usage: ", font=("Arial", 12))
        self.memory_label.pack(pady=10)

        self.swap_label = ttk.Label(root, text="Swap Usage: ", font=("Arial", 12))
        self.swap_label.pack(pady=10)

        self.load_label = ttk.Label(root, text="Load Average: ", font=("Arial", 12))
        self.load_label.pack(pady=10)

        self.iowait_label = ttk.Label(root, text="IO Wait: ", font=("Arial", 12))
        self.iowait_label.pack(pady=10)

        # Start a thread to update metrics
        self.running = True
        self.update_thread = threading.Thread(target=self.update_metrics)
        self.update_thread.start()

    def update_metrics(self):
        """
        Updates the system metrics in the GUI.
        """
        while self.running:
            # Collect metrics
            cpu_usage = psutil.cpu_percent(interval=1)
            memory_usage = psutil.virtual_memory().percent
            swap_usage = psutil.swap_memory().percent
            load_avg = psutil.getloadavg()[0]  # 1-minute load average
            iowait = psutil.cpu_times().iowait

            # Update labels
            self.cpu_label.config(text=f"CPU Usage: {cpu_usage:.2f}%")
            self.memory_label.config(text=f"Memory Usage: {memory_usage:.2f}%")
            self.swap_label.config(text=f"Swap Usage: {swap_usage:.2f}%")
            self.load_label.config(text=f"Load Average: {load_avg:.2f}")
            self.iowait_label.config(text=f"IO Wait: {iowait:.2f}s")

            # Sleep for a short interval to reduce CPU usage
            time.sleep(1)

    def close(self):
        """
        Stops the update thread and closes the window.
        """
        self.running = False
        self.update_thread.join()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = SystemMonitorGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.close)  # Handle window close event
    root.mainloop()