import tkinter as tk
from tkinter import messagebox
import subprocess
import os
import multiprocessing
import psutil
from monitor_interface import SystemMonitorGUI

class KernelTuneGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Kernel Tune Interface")
        self.processes = []

        # Create buttons for each action
        tk.Button(root, text="Simulate CPU Stress", command=self.simulate_cpu_stress).pack(pady=5)
        tk.Button(root, text="Simulate Memory Stress", command=self.simulate_memory_stress).pack(pady=5)
        tk.Button(root, text="Simulate Disk I/O Stress", command=self.simulate_disk_io_stress).pack(pady=5)
        tk.Button(root, text="Open Video", command=self.open_video).pack(pady=5)
        tk.Button(root, text="Ping Network", command=self.ping_network).pack(pady=5)
        tk.Button(root, text="Clean Resources", command=self.clean_resources).pack(pady=5)

        # Add button to open the monitoring window
        tk.Button(root, text="Open System Monitor", command=self.open_monitor).pack(pady=5)

        # Exit button
        tk.Button(root, text="Exit", command=self.exit_application).pack(pady=10)

    def simulate_cpu_stress(self):
        """
        Simulates CPU stress by creating busy loops.
        """
        def _cpu_stress_worker():
            while True:
                pass

        print("Simulating CPU stress...")
        for _ in range(multiprocessing.cpu_count() // 2):  # Use half the cores
            process = multiprocessing.Process(target=_cpu_stress_worker)
            process.start()
            self.processes.append(process)
        messagebox.showinfo("Action", "CPU stress simulation started.")

    def simulate_memory_stress(self):
        """
        Simulates memory stress by allocating large amounts of memory.
        """
        print("Simulating memory stress...")
        self.memory_stress = [" " * 10**6 for _ in range(10**4)]  # Allocate 10MB
        messagebox.showinfo("Action", "Memory stress simulation started.")

    def simulate_disk_io_stress(self):
        """
        Simulates disk I/O stress by creating a large file.
        """
        print("Simulating disk I/O stress...")
        subprocess.run(["dd", "if=/dev/zero", "of=/tmp/largefile", "bs=1M", "count=256"])
        messagebox.showinfo("Action", "Disk I/O stress simulation completed.")

    def open_video(self):
        """
        Opens a video using VLC.
        """
        print("Opening video...")
        video_path = os.path.join(os.getcwd(), "video", "video.mp4")
        try:
            subprocess.Popen(["vlc", "--play-and-exit", video_path], stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
            messagebox.showinfo("Action", "Video opened.")
        except FileNotFoundError:
            messagebox.showerror("Error", "VLC or video file not found.")

    def ping_network(self):
        """
        Simulates network traffic by pinging a server.
        """
        print("Pinging network...")
        subprocess.Popen(["ping", "-c", "4", "8.8.8.8"])
        messagebox.showinfo("Action", "Ping started.")

    def clean_resources(self):
        """
        Cleans up resources created by actions.
        """
        print("Cleaning up resources...")
        for process in self.processes:
            if isinstance(process, multiprocessing.Process):
                process.terminate()
        self.processes = []

        if os.path.exists("/tmp/largefile"):
            os.remove("/tmp/largefile")

        if hasattr(self, "memory_stress"):
            del self.memory_stress

        os.system("pkill stress")
        os.system("pkill yes")
        messagebox.showinfo("Action", "Resources cleaned.")

    def open_monitor(self):
        """
        Opens the system monitoring window.
        """
        monitor_root = tk.Toplevel(self.root)
        SystemMonitorGUI(monitor_root)

    def exit_application(self):
        """
        Exits the application and cleans up resources.
        """
        self.clean_resources()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = KernelTuneGUI(root)
    root.mainloop()