import os
import numpy as np
import psutil
import time

class IoTAgent:
    def __init__(self):
        self.actions = [
            "set_cpu_powersave",
            "set_cpu_ondemand",
            "reduce_writeback_interval",
            "kill_bluetoothd",
            "no_op"
        ]

        self.metrics = [
            "cpu_freq",
            "temperature",
            "disk_write_bytes"
        ]

        self.bins = {
            "cpu_freq": np.linspace(0, 2.5, 5),            # GHz
            "temperature": np.linspace(30, 90, 7),         # °C
            "disk_write_bytes": np.linspace(0, 1e6, 5)     # Bytes
        }

        shape = tuple(len(b) - 1 for b in self.bins.values()) + (len(self.actions),)
        self.q_table = np.zeros(shape)

        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.exploration_rate = 1.0
        self.exploration_decay = 0.995

    def get_state(self):
        freq = psutil.cpu_freq().current / 1000 if psutil.cpu_freq() else 0.0
        temp = self.get_temperature()
        disk_io = psutil.disk_io_counters().write_bytes
        return {
            "cpu_freq": freq,
            "temperature": temp,
            "disk_write_bytes": disk_io
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
        if action == "set_cpu_powersave":
            os.system("sudo sh -c 'echo powersave > /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor'")
        elif action == "set_cpu_ondemand":
            os.system("sudo sh -c 'echo ondemand > /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor'")
        elif action == "reduce_writeback_interval":
            os.system("sudo sysctl -w vm.dirty_writeback_centisecs=1000")
        elif action == "kill_bluetoothd":
            os.system("sudo systemctl stop bluetooth")
        elif action == "no_op":
            pass
        print(f"[ACTION] Executed: {action}")

    def get_temperature(self):
        try:
            temps = psutil.sensors_temperatures()
            if "cpu_thermal" in temps:
                return temps["cpu_thermal"][0].current
            elif "coretemp" in temps:
                return temps["coretemp"][0].current
        except:
            pass
        return 50.0  # Fallback temperature

    def compute_reward(self, state_before, state_after):
        delta_temp = state_before["temperature"] - state_after["temperature"]
        delta_freq = state_before["cpu_freq"] - state_after["cpu_freq"]
        delta_io = state_before["disk_write_bytes"] - state_after["disk_write_bytes"]

        reward = 0
        reward += delta_temp * 0.5
        reward += delta_freq * 1.0
        reward += delta_io / 1e5  # Small reward for less writing

        print(f"[REWARD] ΔTemp={delta_temp:.2f}, ΔFreq={delta_freq:.2f}, ΔDiskIO={delta_io:.0f} → Reward={reward:.2f}")
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
