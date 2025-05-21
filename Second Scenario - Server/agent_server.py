import numpy as np
import psutil
import os
import time

class ServerAgent:
    def __init__(self):
        """
        Initialize the ServerAgent with metric names, actions, bins, and Q-learning parameters.
        """
        self.metric_names = [
            "cpu_usage", "iowait", "interrupts", "ctx_switches", "net_usage", "requests_per_sec"
        ]
        self.state = dict.fromkeys(self.metric_names, 0.0)

        self.actions = [
            "no_op",
            "set_dirty_ratio_10",
            "set_dirty_ratio_40",
            "set_rmem_max_1M",
            "set_rmem_max_16M",
            "set_wmem_max_1M",
            "set_wmem_max_16M",
            "lower_thread_priority",
            "enable_zswap",
            "disable_zswap"
        ]

        self.bins = {
            "cpu_usage": np.linspace(0, 1, 4),
            "iowait": np.linspace(0, 1, 4),
            "interrupts": np.linspace(0, 1, 4),
            "ctx_switches": np.linspace(0, 1, 4),
            "net_usage": np.linspace(0, 1, 4),
            "requests_per_sec": np.linspace(0, 1, 5)
        }
        q_table_shape = tuple(len(b) - 1 for b in self.bins.values()) + (len(self.actions),)
        self.q_table = np.zeros(q_table_shape)

        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.exploration_rate = 1.0
        self.exploration_decay = 0.998  # 0.995

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

    def apply_action(self, action_idx):
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
        elif action == "set_dirty_ratio_40":
            os.system("sudo sysctl -w vm.dirty_ratio=40")
        elif action == "set_rmem_max_1M":
            os.system("sudo sysctl -w net.core.rmem_max=1048576")
        elif action == "set_rmem_max_16M":
            os.system("sudo sysctl -w net.core.rmem_max=16777216")
        elif action == "set_wmem_max_1M":
            os.system("sudo sysctl -w net.core.wmem_max=1048576")
        elif action == "set_wmem_max_16M":
            os.system("sudo sysctl -w net.core.wmem_max=16777216")
        elif action == "lower_thread_priority":
            os.system("sudo renice 10 -p $(pgrep nginx | tr '\n' ' ')")
        elif action == "enable_zswap":
            os.system("sudo sh -c 'echo 1 > /sys/module/zswap/parameters/enabled'")
        elif action == "disable_zswap":
            os.system("sudo sh -c 'echo 0 > /sys/module/zswap/parameters/enabled'")
        return True

    def compute_reward(self, metrics, latency=None, p99=None, debug=False):
        """
        Compute the reward based on system metrics and latency.
        """
        rps = metrics.get("requests_per_sec", 0)
        iowait = metrics.get("iowait", 0)
        cpu = metrics.get("cpu_usage", 0)
        ctx_switches = metrics.get("ctx_switches", 0)
        interrupts = metrics.get("interrupts", 0)

        rps_reward = rps
        if latency is not None:
            latency_factor = max(0, 1 - (latency / 100)) 
            rps_reward *= latency_factor
        
        cpu_penalty = np.exp(cpu - 80) * 10 if cpu > 80 else cpu * 0.5 

        io_penalty = iowait * (100 if iowait > 5 else 10)

        ctx_penalty = np.log1p(ctx_switches) * 0.001

        int_penalty = np.log1p(interrupts) * 0.001

        reward = (
            rps_reward
            - cpu_penalty 
            - io_penalty
            - ctx_penalty 
            - int_penalty
        )

        if debug:
            print("\n=== Reward Calculation Details ===")
            print(f"Base RPS: {rps:.2f}")
            if latency is not None:
                print(f"Latency factor: {latency_factor:.2f}")
                print(f"Adjusted RPS reward: {rps_reward:.2f}")
            print(f"CPU penalty: {cpu_penalty:.2f} (usage: {cpu}%)")
            print(f"IO penalty: {io_penalty:.2f} (iowait: {iowait}%)")
            print(f"Context switches penalty: {ctx_penalty:.2f}")
            print(f"Interrupts penalty: {int_penalty:.2f}")
            print(f"Final reward: {reward:.2f}")
            print("==============================\n")

        return reward

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