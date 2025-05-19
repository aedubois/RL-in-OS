import subprocess
import psutil
import os
import threading
import time
import numpy as np

if os.geteuid() != 0:
    print("Warning: Some actions require root privileges. You may be prompted for your password.")

def get_main_disk():
    """
    Get the main disk device.
    """
    partitions = psutil.disk_partitions()
    for p in partitions:
        if p.mountpoint == '/':
            return p.device
    return '/dev/sda' # Default to /dev/sda if not found

class EventAgent:
    def __init__(self):
        """
        Initialize agent thresholds, state, actions, bins, and Q-table.
        """
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
        self.actions = [
            "reduce_swappiness",
            "increase_dirty_ratio",
            "set_cpu_powersave",
            "set_cpu_performance",
            "lower_process_priority",
            "reduce_io_threads",
            "drop_caches",
            "enable_zswap",
            "increase_read_ahead",
            "kill_stress_processes",
            "clean_tmp",
        ]
        self.bins = {
            "cpu_usage": np.linspace(0, 1, 4),   
            "memory_usage": np.linspace(0, 1, 4), 
            "swap_usage": np.linspace(0, 1, 3),  
            "load_average": np.linspace(0, 1, 4),   
            "disk_usage": np.linspace(0, 1, 4),    
            "temperature": np.linspace(0, 1, 4),   
            "io_wait": np.linspace(0, 1, 3),       
        }
        q_table_shape = tuple(len(bins) - 1 for bins in self.bins.values()) + (len(self.actions),)
        if os.path.exists("First Scenario - Desktop/q_table.npy"):
            self.q_table = np.load("First Scenario - Desktop/q_table.npy")
            print("Q-Table loaded from q_table.npy.")
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
        action_name = self.actions[action_idx]
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
        Normalize state metrics to [0, 1].
        """
        return np.array([
            min(1, self.state["cpu_usage"] / 100),
            min(1, self.state["memory_usage"] / 100),
            min(1, self.state["swap_usage"] / 50),
            min(1, self.state["load_average"] / 10),
            min(1, self.state["disk_usage"] / 100),
            min(1, self.state["temperature"] / 100),
            min(1, self.state["io_wait"] / 20),
        ])

    def discretize_state(self, state):
        """
        Discretize the state into bins for Q-learning.
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
        return tuple(discretized_state)

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
        if action == "reduce_swappiness":
            os.system("sudo sysctl -w vm.swappiness=10")
            reaction = "Swappiness reduced to 10."
        elif action == "increase_dirty_ratio":
            os.system("sudo sysctl -w vm.dirty_ratio=40")
            reaction = "Dirty ratio increased to 40."
        elif action == "set_cpu_powersave":
            os.system("sudo sh -c 'echo powersave > /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor'")
            reaction = "CPU set to powersave mode."
        elif action == "set_cpu_performance":
            os.system("sudo sh -c 'echo performance > /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor'")
            reaction = "CPU set to performance mode."
        elif action == "lower_process_priority":
            os.system("sudo renice +10 -p $(pgrep stress)")
            reaction = "Process priority lowered."
        elif action == "reduce_io_threads":
            os.system("sudo pkill -f 'stress-ng --io'")
            reaction = "Kill stress-ng I/O processes."
        elif action == "drop_caches":
            os.system("sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'")
            reaction = "Caches dropped."
        elif action == "enable_zswap":
            os.system("sudo sh -c 'echo 1 > /sys/module/zswap/parameters/enabled'")
            reaction = "Zswap enabled."
        elif action == "increase_read_ahead":
            disk = get_main_disk()
            os.system(f"sudo blockdev --setra 512 {disk}")
            reaction = f"Read-ahead increased to 512."
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
        delta = state - new_state
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
