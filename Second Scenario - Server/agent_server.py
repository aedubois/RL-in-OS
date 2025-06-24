import numpy as np
import psutil
import os
import time

class ServerAgent:
    def __init__(self, exploration_rate=1.0):
        """
        Initialize the ServerAgent with metric names, actions, bins, and Q-learning parameters.
        """
        self.metric_names = [
            "cpu_usage", "iowait", "interrupts", "ctx_switches", "net_usage", "requests_per_sec"
        ]
        self.state = dict.fromkeys(self.metric_names, 0.0)

        self.actions = [
            "set_dirty_ratio_10",
            "set_dirty_ratio_20",
            "set_dirty_ratio_30",
            "set_dirty_ratio_40",
            "set_rmem_max_1M",
            "set_rmem_max_8M",
            "set_rmem_max_16M",
            "set_wmem_max_1M",
            "set_wmem_max_8M",
            "set_wmem_max_16M",
            "set_tcp_tw_reuse_0",
            "set_tcp_tw_reuse_1",
            "set_tcp_fin_timeout_10",
            "set_tcp_fin_timeout_30",
            "set_somaxconn_128",
            "set_somaxconn_1024",
            "no_op", 
            "reset_rmem_max",
            "reset_wmem_max",
        ]

        self.bins = {
            "cpu_usage": np.linspace(0, 100, 5),         # 0, 25, 50, 75, 100
            "iowait": np.linspace(0, 5, 5),              # 0, 1.25, 2.5, 3.75, 5
            "interrupts": np.linspace(0, 20000, 5),      # 0, 5000, 10000, 15000, 20000
            "ctx_switches": np.linspace(0, 100000, 5),   # 0, 25000, 50000, 75000, 100000
            "net_usage": np.linspace(0, 100, 5),         # 0, 25, 50, 75, 100 (en Mo/s)
            "requests_per_sec": np.linspace(0, 100000, 6) # 0, 20000, 40000, 60000, 80000, 100000
        }
        q_table_shape = tuple(len(b) - 1 for b in self.bins.values()) + (len(self.actions),)
        self.q_table = np.zeros(q_table_shape)

        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.exploration_rate = exploration_rate
        self.exploration_decay = 0.995

        self.last_action_time = {}

    def get_state(self, metrics):
        """
        Normalize and return the state vector from raw metrics.
        """
        state = []
        for name in self.metric_names:
            value = metrics.get(name, 0.0)
            if value is None:
                value = 0.0
            if name == "requests_per_sec":
                norm = min(value / 100000, 1.0)  
            else:
                norm = min(value / 100.0, 1.0)
            state.append(norm)
        return np.array(state)

    def discretize_state(self, state):
        """
        Discretize the normalized state vector into bin indices.
        """
        idx = []
        for i, name in enumerate(self.metric_names):
            bin_idx = np.digitize([state[i]], self.bins[name])[0] - 1
            bin_idx = min(bin_idx, len(self.bins[name]) - 2)
            bin_idx = max(bin_idx, 0)
            idx.append(bin_idx)
        return tuple(idx)

    def select_action(self, state):
        """
        Select an action using epsilon-greedy policy.
        """
        if np.random.rand() < self.exploration_rate:
            return np.random.randint(len(self.actions))
        idx = self.discretize_state(state)
        return np.argmax(self.q_table[idx])

    def learn(self, state, action, reward, new_state):
        """
        Update the Q-table using the Q-learning update rule.
        """
        idx = self.discretize_state(state)
        new_idx = self.discretize_state(new_state)
        best_next = np.max(self.q_table[new_idx])
        td_target = reward + self.discount_factor * best_next
        td_error = td_target - self.q_table[idx][action]
        self.q_table[idx][action] += self.learning_rate * td_error

    def apply_action(self, action_idx, metrics_before=None):
        """
        Apply the selected action to the system.
        """
        action = self.actions[action_idx]
        current_time = time.time()

        if action in self.last_action_time:
            time_since_last = current_time - self.last_action_time[action]
            if time_since_last < 5:
                return False

        self.last_action_time[action] = current_time

        if action == "no_op":
            pass
        elif action == "set_dirty_ratio_10":
            os.system("sudo sysctl -w vm.dirty_ratio=10")
        elif action == "set_dirty_ratio_20":
            os.system("sudo sysctl -w vm.dirty_ratio=20")
        elif action == "set_dirty_ratio_30":
            os.system("sudo sysctl -w vm.dirty_ratio=30")
        elif action == "set_dirty_ratio_40":
            os.system("sudo sysctl -w vm.dirty_ratio=40")
        elif action == "set_rmem_max_1M":
            os.system("sudo sysctl -w net.core.rmem_max=1048576")
        elif action == "set_rmem_max_8M":
            os.system("sudo sysctl -w net.core.rmem_max=8388608")
        elif action == "set_rmem_max_16M":
            os.system("sudo sysctl -w net.core.rmem_max=16777216")
        elif action == "set_wmem_max_1M":
            os.system("sudo sysctl -w net.core.wmem_max=1048576")
        elif action == "set_wmem_max_8M":
            os.system("sudo sysctl -w net.core.wmem_max=8388608")
        elif action == "set_wmem_max_16M":
            os.system("sudo sysctl -w net.core.wmem_max=16777216")
        elif action == "set_tcp_tw_reuse_0":
            os.system("sudo sysctl -w net.ipv4.tcp_tw_reuse=0")
        elif action == "set_tcp_tw_reuse_1":
            os.system("sudo sysctl -w net.ipv4.tcp_tw_reuse=1")
        elif action == "set_tcp_fin_timeout_10":
            os.system("sudo sysctl -w net.ipv4.tcp_fin_timeout=10")
        elif action == "set_tcp_fin_timeout_30":
            os.system("sudo sysctl -w net.ipv4.tcp_fin_timeout=30")
        elif action == "set_somaxconn_128":
            os.system("sudo sysctl -w net.core.somaxconn=128")
        elif action == "set_somaxconn_1024":
            os.system("sudo sysctl -w net.core.somaxconn=1024")
        elif action == "reset_rmem_max":
            os.system("sudo sysctl -w net.core.rmem_max=212992")
        elif action == "reset_wmem_max":
            os.system("sudo sysctl -w net.core.wmem_max=212992")

        time.sleep(0.5)  # Wait for the system to apply the changes


#reward fct 

    def penalize_consecutive_actions(self, current_action, previous_actions, window=3):
        """
        Apply a penalty for consecutive actions that are the same.
        """
        if len(previous_actions) < window:
            return 1.0
        
        recent_actions = previous_actions[-window:]
        if all(a == current_action for a in recent_actions):
            return 0.2 
        
        return 1.0

    def save_q_table(self, path):
        """
        Save the Q-table to a file.
        """
        np.save(path, self.q_table)

    def load_q_table(self, path):
        """
        Load the Q-table from a file.
        """
        self.q_table = np.load(path)
