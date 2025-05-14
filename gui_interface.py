import tkinter as tk
from tkinter import ttk, messagebox
import subprocess
import os
import multiprocessing
import psutil
from monitor_interface import SystemMonitorGUI
from agent import EventAgent

class KernelTuneGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Kernel Tune Interface")
        self.root.geometry("400x700")
        self.processes = []

        # Initialize the EventAgent
        self.agent = EventAgent()

        # Header
        header = ttk.Label(root, text="Kernel Tune Interface", font=("Arial", 16, "bold"))
        header.pack(pady=10)

        # Add button to open the monitoring window (separated at the top)
        ttk.Button(root, text="Open System Monitor", command=self.open_monitor).pack(pady=10)

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

        # Add button to stop all stress (separated at the bottom)
        ttk.Button(root, text="Stop All Stress", command=self.clean_resources).pack(pady=10)

        # Exit button
        ttk.Button(root, text="Exit", command=self.exit_application).pack(pady=10)

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
        self.agent.handle_event("Simulate CPU Stress")
        messagebox.showinfo("Action", "CPU stress simulation started.")

    def simulate_memory_stress(self):
        """
        Simulates memory stress by allocating large amounts of memory.
        """
        print("Simulating memory stress...")
        self.memory_stress = [" " * 10**6 for _ in range(10**4)]  # Allocate 10MB
        self.agent.handle_event("Simulate Memory Stress")
        messagebox.showinfo("Action", "Memory stress simulation started.")

    def simulate_disk_io_stress(self):
        """
        Simulates disk I/O stress by creating a large file.
        """
        print("Simulating disk I/O stress...")
        subprocess.run(["dd", "if=/dev/zero", "of=/tmp/largefile", "bs=1M", "count=256"])
        self.agent.handle_event("Simulate Disk I/O Stress")
        messagebox.showinfo("Action", "Disk I/O stress simulation completed.")

    def simulate_gpu_load(self):
        """
        Simulates GPU load using glxgears or stress-ng.
        """
        print("Simulating GPU load...")
        try:
            subprocess.Popen(["glxgears"], stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
            self.agent.handle_event("Simulate GPU Load")
            messagebox.showinfo("Action", "GPU load simulation started.")
        except FileNotFoundError:
            messagebox.showerror("Error", "glxgears not found. Please install it.")

    def run_compilation_task(self):
        """
        Compiles code.cpp directly to simulate CPU, RAM, and I/O load.
        """
        print("Running compilation task...")
        utils_dir = os.path.join(os.path.dirname(__file__), "utils")
        code_file = os.path.join(utils_dir, "code.cpp")

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

    def simulate_network_flood(self):
        """
        Simulates a network flood using ping.
        """
        print("Simulating network flood...")
        try:
            subprocess.Popen(["ping", "-f", "8.8.8.8"], stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
            self.agent.handle_event("Simulate Network Flood")
            messagebox.showinfo("Action", "Network flood simulation started.")
        except FileNotFoundError:
            messagebox.showerror("Error", "Ping command not found.")

    def fill_disk_until_threshold(self):
        """
        Fills the disk until a certain threshold is reached.
        """
        print("Filling disk until threshold...")
        try:
            subprocess.run(["dd", "if=/dev/zero", "of=/tmp/fillfile", "bs=1M", "count=1024"])
            self.agent.handle_event("Fill Disk Until Threshold")
            messagebox.showinfo("Action", "Disk filling simulation completed.")
        except FileNotFoundError:
            messagebox.showerror("Error", "dd command not found.")

    def play_streaming_video(self):
        """
        Plays a streaming video to simulate network and CPU load.
        """
        print("Playing streaming video...")
        try:
            subprocess.Popen(["vlc", "https://www.youtube.com/watch?v=dQw4w9WgXcQ"], stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
            self.agent.handle_event("Play Streaming Video")
            messagebox.showinfo("Action", "Streaming video started.")
        except FileNotFoundError:
            messagebox.showerror("Error", "VLC not found. Please install it.")

    def spawn_multiple_processes(self):
        """
        Spawns a large number of processes to test the scheduler.
        """
        print("Spawning multiple processes...")
        try:
            for _ in range(100):  # Create 100 processes
                process = subprocess.Popen(["sleep", "100"])
                self.processes.append(process)
            self.agent.handle_event("Spawn Multiple Processes")
            messagebox.showinfo("Action", "Spawned multiple processes.")
        except Exception as e:
            messagebox.showerror("Error", f"An unexpected error occurred:\n{e}")

    def simulate_disk_latency(self):
        """
        Simulates disk latency using stress-ng.
        """
        print("Simulating disk latency...")
        try:
            subprocess.Popen(["stress-ng", "--io", "1", "--timeout", "30"], stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
            self.agent.handle_event("Simulate Disk Latency")
            messagebox.showinfo("Action", "Disk latency simulation started.")
        except FileNotFoundError:
            messagebox.showerror("Error", "stress-ng not found. Please install it.")

    def stress_tmpfs(self):
        """
        Simulates memory pressure by writing to tmpfs (/dev/shm).
        """
        print("Stressing tmpfs...")
        try:
            subprocess.run(["dd", "if=/dev/zero", "of=/dev/shm/tmpfile", "bs=1M", "count=512"])
            self.agent.handle_event("Stress Tmpfs")
            messagebox.showinfo("Action", "Tmpfs stress simulation completed.")
        except FileNotFoundError:
            messagebox.showerror("Error", "dd command not found.")

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

        if os.path.exists("/tmp/fillfile"):
            os.remove("/tmp/fillfile")

        if hasattr(self, "memory_stress"):
            del self.memory_stress

        os.system("pkill stress")
        os.system("pkill yes")
        os.system("pkill glxgears")
        os.system("pkill vlc")
        os.system("pkill ping")
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
