import psutil
import os
import time
import numpy as np
import subprocess

NEGATIVE_ACTIONS_INFO = {
    "simulate_cpu_stress":        1,
    "simulate_memory_stress":     2,
    "simulate_disk_fill":         3,
    "simulate_disk_latency":      2,
    "simulate_network_stress":    2,
}

NEGATIVE_ACTIONS = list(NEGATIVE_ACTIONS_INFO.keys())

def get_negative_action_delay(action):
    return NEGATIVE_ACTIONS_INFO.get(action, 2)

def apply_negative_action(action):
    if action == "simulate_cpu_stress":
        return subprocess.Popen("stress-ng --cpu 2 --timeout 8", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    elif action == "simulate_memory_stress":
        return subprocess.Popen("stress-ng --vm 2 --vm-bytes 1G --timeout 8", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    elif action == "simulate_disk_fill":
        return subprocess.Popen("dd if=/dev/zero of=/tmp/fillfile bs=1M count=1024", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    elif action == "simulate_disk_latency":
        return subprocess.Popen("stress-ng --hdd 2 --hdd-bytes 512M --timeout 8", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    elif action == "simulate_network_stress":
        server_proc = subprocess.Popen("iperf3 -s", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(1)
        client_proc = subprocess.Popen("iperf3 -c 127.0.0.1 -t 8", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return (server_proc, client_proc)
    else:
        return None

def get_main_disk():
    partitions = psutil.disk_partitions()
    for p in partitions:
        if p.mountpoint == '/':
            return p.device
    return '/dev/sda'

def get_param_actions():
    disk = get_main_disk()
    return [
        ("set_dirty_ratio_10", "sudo sysctl -w vm.dirty_ratio=10"),
        ("set_dirty_ratio_40", "sudo sysctl -w vm.dirty_ratio=40"),
        ("set_swappiness_10", "sudo sysctl -w vm.swappiness=10"),
        ("set_swappiness_100", "sudo sysctl -w vm.swappiness=100"),
        ("set_read_ahead_128", f"sudo blockdev --setra 128 {disk}"),
        ("set_read_ahead_1024", f"sudo blockdev --setra 1024 {disk}"),
        ("enable_zswap", "sudo sh -c 'echo 1 > /sys/module/zswap/parameters/enabled'"),
        ("disable_zswap", "sudo sh -c 'echo 0 > /sys/module/zswap/parameters/enabled'"),
    ]

def get_reaction_actions():
    return [
        "drop_caches",
        "kill_stress_processes",
    ]

def get_stress_one_hot(stress_name):
    one_hot = np.zeros(len(NEGATIVE_ACTIONS))
    if stress_name in NEGATIVE_ACTIONS:
        idx = NEGATIVE_ACTIONS.index(stress_name)
        one_hot[idx] = 1
    return one_hot

class LightEventAgent:
    def __init__(self):
        self.state = {
            "cpu_usage": 0,
            "memory_usage": 0,
            "load_average": 0,
            "disk_usage": 0,
            "temperature": 0,
        }
        self.last_stress = None
        self.actions = [a[0] for a in get_param_actions()] + get_reaction_actions()
        self.action_cmds = {a[0]: a[1] for a in get_param_actions()}
        self.bins = {
            "cpu_usage": np.linspace(0, 1, 3), 
            "memory_usage": np.linspace(0, 1, 3),
            "load_average": np.linspace(0, 1, 3),
            "disk_usage": np.linspace(0, 1, 3),
            "temperature": np.linspace(0, 1, 3),
        }
        q_table_shape = tuple(len(bins) - 1 for bins in self.bins.values()) + (len(NEGATIVE_ACTIONS), len(self.actions))
        if os.path.exists("First Scenario - Desktop/light_first_scenario/q_table.npy"):
            self.q_table = np.load("First Scenario - Desktop/light_first_scenario/q_table.npy")
            print("Q-Table loaded from First Scenario - Desktop/light_first_scenario/q_table.npy")
        else:
            self.q_table = np.zeros(q_table_shape)
            print("Initialized new Q-Table.")
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.exploration_rate = 1.0
        self.exploration_decay = 0.995

    def update_metrics_once(self):
        self.state["cpu_usage"] = psutil.cpu_percent(interval=1)
        self.state["memory_usage"] = psutil.virtual_memory().percent
        self.state["load_average"] = os.getloadavg()[0]
        self.state["disk_usage"] = psutil.disk_usage('/').percent
        self.state["temperature"] = self.get_cpu_temperature()

    def get_cpu_temperature(self):
        try:
            temperatures = psutil.sensors_temperatures()
            if "coretemp" in temperatures:
                core_temps = temperatures["coretemp"]
                if core_temps:
                    return core_temps[0].current
        except Exception:
            pass
        return 0

    def get_normalized_state(self):
        base = np.array([
            min(1, self.state["cpu_usage"] / 100),
            min(1, self.state["memory_usage"] / 100),
            min(1, self.state["load_average"] / 10),
            min(1, self.state["disk_usage"] / 100),
            min(1, self.state["temperature"] / 100),
        ])
        stress_one_hot = get_stress_one_hot(self.last_stress) if self.last_stress else np.zeros(len(NEGATIVE_ACTIONS))
        return np.concatenate([base, stress_one_hot])

    def discretize_state(self, state):
        metrics_order = [
            "cpu_usage",
            "memory_usage",
            "load_average",
            "disk_usage",
            "temperature",
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

    def select_action(self, state):
        discretized_state = self.discretize_state(state)
        if np.random.uniform(0, 1) < self.exploration_rate:
            return np.random.randint(0, len(self.actions))
        else:
            return np.argmax(self.q_table[discretized_state])

    def apply_action(self, action_idx):
        action = self.actions[action_idx]
        if action in self.action_cmds:
            os.system(self.action_cmds[action])
        elif action == "drop_caches":
            os.system("sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'")
        elif action == "kill_stress_processes":
            os.system("sudo pkill -f stress-ng")
            os.system("sudo pkill -f yes")
            os.system("sudo pkill -f vlc")
            os.system("sudo pkill -f iperf3")
        else:
            print(f"Unknown action: {action}")

    def compute_reward(self, state, new_state, debug=False):
        weights = np.array([1.0, 0.6, 0.8, 0.4, 0.9])
        delta = state[:5] - new_state[:5]
        affected = np.abs(delta) > 0.01
        reward = np.sum(delta[affected] * weights[affected]) * 10
        if not np.any(affected):
            reward -= 1
        if debug:
            metric_names = ["CPU", "RAM", "Load", "Disk", "Temp"]
            print("\nReward details:")
            for i, name in enumerate(metric_names):
                if affected[i]:
                    print(f"  {name}: Δ={delta[i]:+.3f} weight={weights[i]} → +{delta[i]*weights[i]*10:.2f}")
                else:
                    print(f"  {name}: Δ={delta[i]:+.3f} (ignored)")
            print(f"Total reward: {reward:.2f}")
        return reward

    def learn(self, state, action, reward, new_state):
        state_idx = self.discretize_state(state)
        new_state_idx = self.discretize_state(new_state)
        self.q_table[state_idx][action] = self.q_table[state_idx][action] + self.learning_rate * (
            reward + self.discount_factor * np.max(self.q_table[new_state_idx]) - self.q_table[state_idx][action]
        )

    def save_q_table(self, path):
        np.save(path, self.q_table)
        print(f"Q-Table saved to {path}.")

    def clean_resources(self):
        for path in ["/tmp/largefile", "/tmp/fillfile", "/dev/shm/tmpfs_stress"]:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except Exception as e:
                    print(f"Error removing {path}: {e}")
        os.system("pkill -f stress-ng")
        os.system("pkill -f stress")
        os.system("pkill -f yes")
        os.system("pkill -f vlc")
        os.system("pkill -f iperf3")