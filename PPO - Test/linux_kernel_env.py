import psutil
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os
import subprocess
import multiprocessing
import time


class KernelTuneEnv(gym.Env):
    """
    Environment for tuning Linux kernel parameters and observing the effects on system metrics.
    """
    def __init__(self):
        super(KernelTuneEnv, self).__init__()
        self.action_space = spaces.Discrete(7)  # 7 possible actions (including corrective actions)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(6,), dtype=np.float32)
        self.state = None
        self.previous_state = None
        self.step_count = 0
        self.max_steps = 1000
        self.processes = []  # To track processes created by actions

    def reset(self, seed=None, options=None):
        """
        Resets the environment to its initial state.
        """
        super().reset(seed=seed)
        self.clean_resources()  # Clean up any leftover resources
        self.state = self._get_initial_state()
        self.previous_state = self.state.copy()
        self.step_count = 0
        return self.state, {}

    def step(self, action):
        """
        Applies an action and returns the next state, reward, and episode termination indicators.
        """
        self.previous_state = self.state.copy()  # Save the previous state
        self._apply_action(action)
        self.state = self._get_next_state()
        reward = self._calculate_reward()
        self.step_count += 1
        terminated = self.step_count >= self.max_steps
        truncated = False
        return self.state, reward, terminated, truncated, {}

    def _get_initial_state(self):
        """
        Generates the initial state of the environment based on real system metrics.
        """
        return self._collect_system_metrics()

    def _apply_action(self, action):
        """
        Applies an action to influence the metrics.
        """
        if action == 0:
            self._simulate_network_traffic()
        elif action == 1:
            self._create_large_file()
        elif action == 2:
            self._play_video()
        elif action == 3:
            self._cpu_stress()
        elif action == 4:
            self._memory_stress()
        elif action == 5:
            self._disk_io_stress()
        elif action == 6:  # Corrective action to stop stress processes
            self._stop_stress()

    def _get_next_state(self):
        """
        Collects the next state based on real system metrics.
        """
        return self._collect_system_metrics()

    def _collect_system_metrics(self):
        """
        Collects real system metrics using psutil.
        """
        cpu_usage = psutil.cpu_percent(interval=1.0) / 100.0  # Normalized to [0, 1]
        ram_usage = psutil.virtual_memory().percent / 100.0  # Normalized to [0, 1]
        swap_usage = psutil.swap_memory().percent / 100.0  # Normalized to [0, 1]
        iowait = psutil.cpu_times().iowait / 100.0  # Normalized I/O wait time
        load_avg = psutil.getloadavg()[0] / psutil.cpu_count()  # Load average per core
        latency = np.random.uniform(0.0, 1.0)  # Placeholder for network latency

        return np.array([cpu_usage, swap_usage, load_avg, ram_usage, iowait, latency], dtype=np.float32)

    def _calculate_reward(self):
        """
        Calculates the reward based on defined objectives with tolerance zones.
        """
        # Targets and tolerances
        cpu_target = 0.4  
        ram_target = 0.35
        load_target = 0.2

        # Penalties
        cpu_penalty = max(0, abs(self.state[0] - cpu_target) - 0.1)
        ram_penalty = max(0, abs(self.state[3] - ram_target) - 0.1)
        load_penalty = max(0, abs(self.state[2] - load_target) - 0.1)

        # Bonus for reducing CPU or RAM usage
        cpu_bonus = max(0, self.previous_state[0] - self.state[0])  # Reduction in CPU usage
        ram_bonus = max(0, self.previous_state[3] - self.state[3])  # Reduction in RAM usage

        # Reward calculation
        reward = 1.0 - (0.3 * cpu_penalty + 0.2 * ram_penalty + 0.1 * load_penalty) + 0.1 * (cpu_bonus + ram_bonus)

        # Ensure reward is within [0, 1]
        return max(reward, 0.0)

    def _simulate_network_traffic(self):
        """
        Simulates network traffic using ping.
        """
        process = subprocess.Popen(["ping", "-c", "10", "8.8.8.8"])
        self.processes.append(process)

    def _create_large_file(self):
        """
        Creates a large file to simulate disk usage.
        """
        subprocess.run(["dd", "if=/dev/zero", "of=/tmp/largefile", "bs=1M", "count=1024"])

    def _play_video(self):
        """
        Simulates playing a video using VLC without opening a video window.
        """
        try:
            video_path = os.path.join(os.getcwd(), "video", "video.mp4")
            process = subprocess.Popen(
                ["vlc", "--intf", "dummy", "--play-and-exit", "--run-time=2", "--no-video", video_path],
                stderr=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL
            )
            process.wait()  # Wait for VLC to finish
        except FileNotFoundError:
            print("Error: VLC or the video file is missing.")

    def _cpu_stress(self):
        """
        Simulates CPU stress by creating busy loops.
        """
        def stress():
            while True:
                pass
        for _ in range(multiprocessing.cpu_count() // 2):  # Use only half the cores
            process = multiprocessing.Process(target=stress)
            process.start()
            self.processes.append(process)

    def _memory_stress(self):
        """
        Simulates memory stress by allocating large amounts of memory progressively.
        """
        self.memory_stress = [" " * 10**6 for _ in range(10**4)]  # Allocate 10MB of memory

    def _disk_io_stress(self):
        """
        Simulates disk I/O stress by reading a large file repeatedly.
        """
        if not os.path.exists("/tmp/largefile"):
            subprocess.run(["dd", "if=/dev/zero", "of=/tmp/largefile", "bs=1M", "count=1024"])
        process = subprocess.Popen(["cat", "/tmp/largefile"])
        self.processes.append(process)

    def _stop_stress(self):
        """
        Stops all stress processes to reduce system load.
        """
        print("Stopping stress processes...")
        self.clean_resources()

    def clean_resources(self):
        """
        Cleans up resources created by actions.
        """
        # Terminate all processes
        for process in self.processes:
            if isinstance(process, subprocess.Popen):
                process.terminate()
            elif isinstance(process, multiprocessing.Process):
                process.terminate()
        self.processes = []

        # Remove large files
        if os.path.exists("/tmp/largefile"):
            os.remove("/tmp/largefile")

        # Clear memory stress
        if hasattr(self, "memory_stress"):
            del self.memory_stress

    def render(self):
        """
        Displays the current state of the environment.
        """
        print(f"Current state: {self.state}")

    def close(self):
        """
        Cleans up resources used by the environment.
        """
        self.clean_resources()
