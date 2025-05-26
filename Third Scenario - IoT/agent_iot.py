import os
import numpy as np
import time
import random

class IoTAgent:
    def __init__(self):
        self.actions = [
            "set_cpu_powersave",
            "set_cpu_ondemand",
            "reduce_writeback_interval",
            "enable_sleep_mode",
            "reduce_screen_brightness",
            "no_op"
        ]

        self.metrics = [
            "cpu_freq",
            "temperature",
            "battery",
            "disk_write_bytes",
            "error_rate",
            "network_usage"
        ]

        self.bins = {
            "cpu_freq": np.linspace(0.8, 2.5, 5),            # GHz
            "temperature": np.linspace(20, 80, 7),           # °C
            "battery": np.linspace(0, 100, 6),               # %
            "disk_write_bytes": np.linspace(0, 1e7, 5),      # Bytes
            "error_rate": np.linspace(0, 1, 5),              # 0-1 (fraction)
            "network_usage": np.linspace(0, 1e6, 5)          # Bytes/sec
        }

        shape = tuple(len(b) - 1 for b in self.bins.values()) + (len(self.actions),)
        self.q_table = np.zeros(shape)

        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.exploration_rate = 1.0
        self.exploration_decay = 0.995

        # Simulated state
        self.sim_cpu_freq = 2.0
        self.sim_temperature = 40.0
        self.sim_battery = 100.0
        self.sim_disk_io = 1e6
        self.sim_error_rate = 0.01
        self.sim_network_usage = 1e4

        self.sleep_mode_steps = 0

    def reset(self):
        self.sim_cpu_freq = 2.0
        self.sim_temperature = 40.0
        self.sim_battery = 100.0
        self.sim_disk_io = 1e6
        self.sim_error_rate = 0.01
        self.sim_network_usage = 1e4
        self.sleep_mode_steps = 0
        return self.get_state()

    def get_state(self):
        return {
            "cpu_freq": self.sim_cpu_freq,
            "temperature": self.sim_temperature,
            "battery": self.sim_battery,
            "disk_write_bytes": self.sim_disk_io,
            "error_rate": self.sim_error_rate,
            "network_usage": self.sim_network_usage
        }

    def normalize_state(self, raw_state):
        normalized = []
        for key in self.metrics:
            value = raw_state[key]
            bin_edges = self.bins[key]
            bin_idx = np.digitize([value], bin_edges)[0] - 1
            bin_idx = max(0, min(bin_idx, len(bin_edges) - 2))
            normalized.append(bin_idx)
        return tuple(normalized)

    def select_action(self, state_tuple):
        if self.sleep_mode_steps > 0:
            return self.actions.index("no_op")
        if np.random.rand() < self.exploration_rate:
            return np.random.randint(len(self.actions))
        return np.argmax(self.q_table[state_tuple])

    def learn(self, state, action_idx, reward, next_state):
        s = self.normalize_state(state)
        s_prime = self.normalize_state(next_state)
        best_next = np.max(self.q_table[s_prime])
        td_target = reward + self.discount_factor * best_next
        td_error = td_target - self.q_table[s][action_idx]
        self.q_table[s][action_idx] += self.learning_rate * td_error

    def apply_action(self, action_idx):
        action = self.actions[action_idx]
        print(f"[ACTION] Executed: {action}")

        # Simulate action effects
        if self.sleep_mode_steps > 0:
            self.sleep_mode_steps -= 1
            # In sleep mode, everything cools down and battery is saved
            self.sim_cpu_freq = 0.8
            self.sim_temperature = max(20, self.sim_temperature - 3)
            self.sim_battery = min(100, self.sim_battery + 1)
            self.sim_disk_io = max(0, self.sim_disk_io - 1e5)
            self.sim_error_rate = max(0, self.sim_error_rate - 0.05)
            self.sim_network_usage = max(0, self.sim_network_usage - 1e5)
        else:
            if action == "set_cpu_powersave":
                self.sim_cpu_freq = max(0.8, self.sim_cpu_freq - 0.3)
                self.sim_temperature = max(20, self.sim_temperature - 2)
                self.sim_battery = min(100, self.sim_battery + 0.5)
            elif action == "set_cpu_ondemand":
                self.sim_cpu_freq = min(2.5, self.sim_cpu_freq + 0.3)
                self.sim_temperature = min(80, self.sim_temperature + 2)
                self.sim_battery = max(0, self.sim_battery - 1)
            elif action == "reduce_writeback_interval":
                self.sim_disk_io = max(0, self.sim_disk_io - 2e5)
                self.sim_battery = min(100, self.sim_battery + 0.2)
            elif action == "enable_sleep_mode":
                self.sleep_mode_steps = 3  # Sleep for 3 steps
            elif action == "reduce_screen_brightness":
                self.sim_battery = min(100, self.sim_battery + 0.7)
                self.sim_temperature = max(20, self.sim_temperature - 0.5)
                self.sim_error_rate = max(0, self.sim_error_rate - 0.01)
            # no_op does nothing

        # Random event: load spike
        if random.random() < 0.1:
            print("[EVENT] Load spike!")
            self.sim_temperature += random.uniform(2, 6)
            self.sim_disk_io += random.randint(int(1e5), int(5e5))
            self.sim_error_rate = min(1.0, self.sim_error_rate + random.uniform(0.05, 0.2))
            self.sim_network_usage += random.randint(int(1e5), int(5e5))

        # Natural evolution
        self.sim_temperature += np.random.uniform(-0.5, 1.0)
        self.sim_battery = max(0, self.sim_battery - np.random.uniform(0.1, 0.5))
        self.sim_disk_io += np.random.randint(int(1e4), int(1e5))
        self.sim_error_rate = min(1.0, max(0, self.sim_error_rate + np.random.uniform(-0.01, 0.02)))
        self.sim_network_usage = max(0, self.sim_network_usage + np.random.randint(-int(1e4), int(5e4)))

    def compute_reward(self, state_before, state_after):
        delta_temp = state_after["temperature"] - state_before["temperature"]
        delta_battery = state_after["battery"] - state_before["battery"]
        delta_disk = state_after["disk_write_bytes"] - state_before["disk_write_bytes"]
        delta_error = state_after["error_rate"] - state_before["error_rate"]
        delta_net = state_after["network_usage"] - state_before["network_usage"]

        # Reward: encourage low temp, high battery, low disk I/O, low error, low network usage
        reward = (
            -delta_temp
            + delta_battery / 2
            - abs(delta_disk) / 1e6
            - 10 * delta_error
            - abs(delta_net) / 1e5
        )
        print(f"[REWARD] ΔTemp={delta_temp:.2f}, ΔBattery={delta_battery:.2f}, ΔDiskIO={delta_disk:.0f}, ΔError={delta_error:.3f}, ΔNet={delta_net:.0f} → Reward={reward:.2f}")
        return reward

    def save_q_table(self, path="q_table_iot.npy"):
        np.save(path, self.q_table)

    def load_q_table(self, path="q_table_iot.npy"):
        if os.path.exists(path):
            self.q_table = np.load(path)
            print("[Q-TABLE] Loaded from file.")

if __name__ == "__main__":
    agent = IoTAgent()
    for _ in range(3):
        state = agent.get_state()
        state_tuple = agent.normalize_state(state)
        action = agent.select_action(state_tuple)
        agent.apply_action(action)
        time.sleep(2)
        next_state = agent.get_state()
        reward = agent.compute_reward(state, next_state)
        agent.learn(state, action, reward, next_state)
    agent.save_q_table()
