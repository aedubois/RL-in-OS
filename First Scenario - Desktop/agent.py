import psutil
import os
import threading
import time
import numpy as np
import subprocess

NEGATIVE_ACTIONS_INFO = {
    "simulate_cpu_stress":        1,
    "simulate_memory_stress":     2,
    "simulate_disk_fill":         3,
    "simulate_disk_latency":      2,
    "stress_tmpfs":               2,
    "play_streaming_video":       4,
    "simulate_network_stress":    2,
    "simulate_swap_stress":       3,
    "simulate_high_load":         1,
    "simulate_temp_increase":     2,
    "no_op":                      0,
}

NEGATIVE_ACTIONS = list(NEGATIVE_ACTIONS_INFO.keys())

def get_negative_action_delay(action):
    return NEGATIVE_ACTIONS_INFO.get(action, 2)

def apply_negative_action(action):
    if action == "no_op":
        print("No operation performed.")
        return None
    if action == "simulate_cpu_stress":
        print("Simulating CPU stress...")
        return subprocess.Popen("stress-ng --cpu 2 --timeout 8", shell=True)
    elif action == "simulate_memory_stress":
        print("Simulating memory stress...")
        return subprocess.Popen("stress-ng --vm 2 --vm-bytes 1G --timeout 8", shell=True)
    elif action == "simulate_disk_fill":
        print("Simulating disk fill...")
        return subprocess.Popen("dd if=/dev/zero of=/tmp/fillfile bs=1M count=2048", shell=True)
    elif action == "simulate_disk_latency":
        print("Simulating disk latency...")
        return subprocess.Popen("stress-ng --hdd 2 --hdd-bytes 1G --timeout 8", shell=True)
    elif action == "stress_tmpfs":
        print("Simulating tmpfs stress...")
        return subprocess.Popen("dd if=/dev/zero of=/dev/shm/tmpfs_stress bs=1M count=1024", shell=True)
    elif action == "play_streaming_video":
        print("Playing streaming video...")
        return subprocess.Popen(
            "vlc --intf dummy --run-time=8 --play-and-exit https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4 vlc://quit",
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
    elif action == "simulate_network_stress":
        print("Simulating network stress...")
        server_proc = subprocess.Popen("iperf3 -s", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(1)
        client_proc = subprocess.Popen("iperf3 -c 127.0.0.1 -t 8", shell=True)
        return (server_proc, client_proc)
    elif action == "simulate_swap_stress":
        print("Simulating swap stress...")
        return subprocess.Popen("sudo stress-ng --swap 2 --timeout 8", shell=True)
    elif action == "simulate_high_load":
        print("Simulating high load...")
        return subprocess.Popen("timeout 8s yes > /dev/null", shell=True)
    elif action == "simulate_temp_increase":
        print("Simulating temperature increase...")
        return subprocess.Popen("stress-ng --cpu 4 --timeout 8", shell=True)
    else:
        return None

def get_main_disk():
    """
    Get the main disk device.
    """
    partitions = psutil.disk_partitions()
    for p in partitions:
        if p.mountpoint == '/':
            return p.device
    return '/dev/sda'

def get_param_actions():
    disk = get_main_disk()
    return [
        # dirty_ratio
        ("set_dirty_ratio_10", "sudo sysctl -w vm.dirty_ratio=10"),
        ("set_dirty_ratio_20", "sudo sysctl -w vm.dirty_ratio=20"),
        ("set_dirty_ratio_40", "sudo sysctl -w vm.dirty_ratio=40"),
        # swappiness
        ("set_swappiness_10", "sudo sysctl -w vm.swappiness=10"),
        ("set_swappiness_60", "sudo sysctl -w vm.swappiness=60"),
        ("set_swappiness_100", "sudo sysctl -w vm.swappiness=100"),
        # read_ahead 
        ("set_read_ahead_128", f"sudo blockdev --setra 128 {disk}"),
        ("set_read_ahead_512", f"sudo blockdev --setra 512 {disk}"),
        ("set_read_ahead_1024", f"sudo blockdev --setra 1024 {disk}"),
        # cpu governor
        ("set_cpu_powersave", "sudo sh -c 'echo powersave > /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor'"),
        ("set_cpu_performance", "sudo sh -c 'echo performance > /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor'"),
        # zswap
        ("enable_zswap", "sudo sh -c 'echo 1 > /sys/module/zswap/parameters/enabled'"),
        ("disable_zswap", "sudo sh -c 'echo 0 > /sys/module/zswap/parameters/enabled'"),
    ]

def get_reaction_actions():
    return [
        "lower_process_priority",
        "reduce_io_threads",
        "drop_caches",
        "kill_stress_processes",
        "clean_tmp",
        "no_op", 
    ]

def get_stress_one_hot(stress_name):
    one_hot = np.zeros(len(NEGATIVE_ACTIONS))
    if stress_name in NEGATIVE_ACTIONS:
        idx = NEGATIVE_ACTIONS.index(stress_name)
        one_hot[idx] = 1
    return one_hot
class EventAgent:
    def __init__(self):
        """ Initialize the EventAgent with system metrics and thresholds."""
        self.thresholds = {
            "high_cpu": 80,
            "high_memory": 80,
            "high_temperature": 80,
            "low_disk_space": 10,
        }
        self.state = {
            "cpu_usage": 0,
            "memory_usage": 0,
            "swap_usage": 0,
            "load_average": 0,
            "disk_usage": 0,
            "temperature": 0,
            "io_wait": 0,
        }
        self.last_stress = None
        self.actions = [a[0] for a in get_param_actions()] + get_reaction_actions()
        self.action_cmds = {a[0]: a[1] for a in get_param_actions()}
        self.bins = {
            "cpu_usage": np.linspace(0, 1, 4),
            "memory_usage": np.linspace(0, 1, 4),
            "swap_usage": np.linspace(0, 1, 3),
            "load_average": np.linspace(0, 1, 4),
            "disk_usage": np.linspace(0, 1, 4),
            "temperature": np.linspace(0, 1, 4),
            "io_wait": np.linspace(0, 1, 3),
        }
        q_table_shape = tuple(len(bins) - 1 for bins in self.bins.values()) + (len(NEGATIVE_ACTIONS), len(self.actions))
        if os.path.exists("First Scenario - Desktop/q_table.npy"):
            self.q_table = np.load("First Scenario - Desktop/q_table.npy")
            print("Q-Table loaded from  First Scenario - Desktop/q_table.npy")
        else:
            self.q_table = np.zeros(q_table_shape)
            print("Initialized new Q-Table.")
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.exploration_rate = 1.0
        self.exploration_decay = 0.995
        self.running = True

    def monitor_metrics(self):
        """
        Monitor system metrics and update state.
        """
        while self.running:
            try:
                self.state["cpu_usage"] = psutil.cpu_percent(interval=1)
                self.state["memory_usage"] = psutil.virtual_memory().percent
                self.state["swap_usage"] = psutil.swap_memory().percent
                self.state["load_average"] = os.getloadavg()[0]
                self.state["disk_usage"] = psutil.disk_usage('/').percent
                self.state["temperature"] = self.get_cpu_temperature()
                self.state["io_wait"] = psutil.cpu_times().iowait if hasattr(psutil.cpu_times(), 'iowait') else 0
                self.check_thresholds()
            except Exception as e:
                print(f"Error monitoring metrics: {e}")

    def get_cpu_temperature(self):
        """
        Get CPU temperature if available.
        """
        try:
            temperatures = psutil.sensors_temperatures()
            if "coretemp" in temperatures:
                core_temps = temperatures["coretemp"]
                if core_temps:
                    return core_temps[0].current
        except Exception as e:
            print(f"Error getting CPU temperature: {e}")
        return 0

    def check_thresholds(self):
        """
        Check if any metrics exceed thresholds and trigger events.
        """
        if self.state["cpu_usage"] > self.thresholds["high_cpu"]:
            self.handle_event("High CPU Usage")
        if self.state["memory_usage"] > self.thresholds["high_memory"]:
            self.handle_event("High Memory Usage")
        if self.state["temperature"] > self.thresholds["high_temperature"]:
            self.handle_event("High Temperature")
        if 100 - self.state["disk_usage"] < self.thresholds["low_disk_space"]:
            self.handle_event("Low Disk Space")

    def handle_event(self, event_type, plot=False):
        """
        Handle events triggered by GUI or thresholds.
        If plot=True, return the action description for display.
        """
        print(f"Event received: {event_type}")
        state = self.get_normalized_state()
        action_idx = self.select_action(state)
        reaction_text = self.apply_action(action_idx, return_text=plot)
        time.sleep(1)
        self.update_metrics_once()
        new_state = self.get_normalized_state()
        reward = self.compute_reward(state, new_state)
        self.learn(state, action_idx, reward, new_state)
        if plot:
            return reaction_text

    def update_metrics_once(self):
        """
        Update state with current system metrics.
        """
        self.state["cpu_usage"] = psutil.cpu_percent(interval=1)
        self.state["memory_usage"] = psutil.virtual_memory().percent
        self.state["swap_usage"] = psutil.swap_memory().percent
        self.state["load_average"] = os.getloadavg()[0]
        self.state["disk_usage"] = psutil.disk_usage('/').percent
        self.state["temperature"] = self.get_cpu_temperature()
        self.state["io_wait"] = psutil.cpu_times().iowait if hasattr(psutil.cpu_times(), 'iowait') else 0

    def get_normalized_state(self):
        """
        Normalize state metrics to [0, 1] and concat one-hot stress.
        """
        base = np.array([
            min(1, self.state["cpu_usage"] / 100),
            min(1, self.state["memory_usage"] / 100),
            min(1, self.state["swap_usage"] / 50),
            min(1, self.state["load_average"] / 10),
            min(1, self.state["disk_usage"] / 100),
            min(1, self.state["temperature"] / 100),
            min(1, self.state["io_wait"] / 20),
        ])
        stress_one_hot = get_stress_one_hot(self.last_stress) if self.last_stress else np.zeros(len(NEGATIVE_ACTIONS))
        return np.concatenate([base, stress_one_hot])

    def discretize_state(self, state):
        """
        Discretize the state into bins for Q-learning, including stress.
        """
        metrics_order = [
            "cpu_usage",
            "memory_usage",
            "swap_usage",
            "load_average",
            "disk_usage",
            "temperature",
            "io_wait",
        ]
        discretized_state = []
        for i, metric in enumerate(metrics_order):
            value = state[i]
            bin_edges = self.bins[metric]
            bin_idx = np.digitize(value, bin_edges) - 1
            bin_idx = max(0, min(bin_idx, len(bin_edges) - 2))
            discretized_state.append(bin_idx)
        if self.last_stress and self.last_stress in NEGATIVE_ACTIONS:
            stress_idx = NEGATIVE_ACTIONS.index(self.last_stress)
        else:
            stress_idx = 0
        discretized_state.append(stress_idx)
        return tuple(discretized_state)
    
    def reset_all_params(self):
        """
        Reset all system parameters to default values.
        """
        disk = get_main_disk()
        subprocess.run("sudo sysctl -w vm.dirty_ratio=20", shell=True)
        subprocess.run("sudo sysctl -w vm.swappiness=60", shell=True)
        subprocess.run(f"sudo blockdev --setra 128 {disk}", shell=True)
        subprocess.run("sudo sh -c 'echo performance > /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor'", shell=True)
        subprocess.run("sudo sh -c 'echo 0 > /sys/module/zswap/parameters/enabled'", shell=True)
        time.sleep(1)

    def select_action(self, state):
        """
        Select an action based on the current state using epsilon-greedy policy
        """
        discretized_state = self.discretize_state(state)
        if np.random.uniform(0, 1) < self.exploration_rate:
            return np.random.randint(0, len(self.actions))
        else:
            return np.argmax(self.q_table[discretized_state])

    def apply_action(self, action_idx, return_text=False):
        """
        Apply the selected action to the system.
        If return_text is True, return a description of the action.
        """
        action = self.actions[action_idx]
        reaction = ""
        if action == "no_op":
            reaction = "No operation performed."
        elif action in self.action_cmds:
            os.system(self.action_cmds[action])
            reaction = f"{action.replace('_', ' ').capitalize()} applied."
        elif action == "lower_process_priority":
            os.system("sudo renice +10 -p $(pgrep stress)")
            reaction = "Process priority lowered."
        elif action == "reduce_io_threads":
            os.system("sudo pkill -f 'stress-ng --io'")
            reaction = "Kill stress-ng I/O processes."
        elif action == "drop_caches":
            os.system("sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'")
            reaction = "Caches dropped."
        elif action == "kill_stress_processes":
            os.system("sudo pkill -f stress-ng")
            os.system("sudo pkill -f yes")
            os.system("sudo pkill -f vlc")
            os.system("sudo pkill -f iperf3")
            reaction = "All stress processes killed."
        elif action == "clean_tmp":
            os.system("sudo rm -rf /tmp/*")
            reaction = "Temporary files cleaned."
        else:
            reaction = f"Unknown action: {action}"
        print(reaction)
        if return_text:
            return reaction

    def compute_reward(self, state, new_state, debug=False):
        """
        Compute reward based on improvements between previous and current state.
        """
        weights = np.array([1.0, 0.6, 0.8, 0.4, 0.3, 0.9, 0.5])
        delta = state[:7] - new_state[:7]
        affected = np.abs(delta) > 0.01
        reward = np.sum(delta[affected] * weights[affected]) * 10
        if not np.any(affected):
            reward -= 1  
        if debug:
            metric_names = ["CPU", "RAM", "SWAP", "Load", "Disk", "Temp", "IOwait"]
            print("\nReward details:")
            for i, name in enumerate(metric_names):
                if affected[i]:
                    print(f"  {name}: Δ={delta[i]:+.3f} weight={weights[i]} → +{delta[i]*weights[i]*10:.2f}")
                else:
                    print(f"  {name}: Δ={delta[i]:+.3f} (ignored)")
            print(f"Total reward: {reward:.2f}")
        return reward

    def learn(self, state, action, reward, new_state):
        """
        Update Q-table using the Q-learning algorithm.
        """
        state_idx = self.discretize_state(state)
        new_state_idx = self.discretize_state(new_state)
        self.q_table[state_idx][action] = self.q_table[state_idx][action] + self.learning_rate * (
            reward + self.discount_factor * np.max(self.q_table[new_state_idx]) - self.q_table[state_idx][action]
        )

    def stop(self):
        """
        Stop the agent's monitoring thread.
        """
        self.running = False

    def save_q_table(self, path):
        np.save(path, self.q_table)
        print(f"Q-Table saved to {path}.")

    def clean_resources(self):
        """
        Clean up resources and processes.
        """
        if hasattr(self, "processes"):
            for process in self.processes:
                try:
                    if hasattr(process, "terminate"):
                        process.terminate()
                except Exception as e:
                    print(f"Error terminating process: {e}")
            self.processes = []
        # Remove temporary files
        for path in ["/tmp/largefile", "/tmp/fillfile", "/dev/shm/tmpfs_stress"]:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except Exception as e:
                    print(f"Error removing {path}: {e}")
        if hasattr(self, "memory_stress"):
            del self.memory_stress
        # Kill any remaining stress processes
        os.system("pkill -f stress-ng")
        os.system("pkill -f stress")
        os.system("pkill -f yes")
        os.system("pkill -f glxgears")
        os.system("pkill -f vlc")
        os.system("pkill -f iperf3")
        os.system("pkill -f ping")

if __name__ == "__main__":
    agent = EventAgent()
    monitoring_thread = threading.Thread(target=agent.monitor_metrics, daemon=True)
    monitoring_thread.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        agent.stop()
