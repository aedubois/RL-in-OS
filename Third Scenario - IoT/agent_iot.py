import os
import numpy as np
import time
import random

class IoTAgent:
    def __init__(self):
        """Initialize the IoT agent with Q-learning parameters and state space."""
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
        """Reset the simulated environment to initial state."""
        self.sim_cpu_freq = 2.0
        self.sim_temperature = 40.0
        self.sim_battery = 100.0
        self.sim_disk_io = 1e6
        self.sim_error_rate = 0.01
        self.sim_network_usage = 1e4
        self.sleep_mode_steps = 0
        return self.get_state()

    def get_state(self):
        """Get the current state of the simulated IoT device."""
        return {
            "cpu_freq": self.sim_cpu_freq,
            "temperature": self.sim_temperature,
            "battery": self.sim_battery,
            "disk_write_bytes": self.sim_disk_io,
            "error_rate": self.sim_error_rate,
            "network_usage": self.sim_network_usage
        }

    def normalize_state(self, raw_state):
        """Normalize the raw state into a tuple of indices based on defined bins."""
        normalized = []
        for key in self.metrics:
            value = raw_state[key]
            bin_edges = self.bins[key]
            bin_idx = np.digitize([value], bin_edges)[0] - 1
            bin_idx = max(0, min(bin_idx, len(bin_edges) - 2))
            normalized.append(bin_idx)
        return tuple(normalized)

    def select_action(self, state_tuple):
        """Select an action based on the current state using epsilon-greedy policy."""
        if self.sleep_mode_steps > 0:
            return self.actions.index("no_op")
        if np.random.rand() < self.exploration_rate:
            return np.random.randint(len(self.actions))
        return np.argmax(self.q_table[state_tuple])

    def learn(self, state, action_idx, reward, next_state):
        """Update the Q-table based on the action taken and the received reward."""
        s = self.normalize_state(state)
        s_prime = self.normalize_state(next_state)
        best_next = np.max(self.q_table[s_prime])
        td_target = reward + self.discount_factor * best_next
        td_error = td_target - self.q_table[s][action_idx]
        self.q_table[s][action_idx] += self.learning_rate * td_error

    def load_spikes(self):
        """Simulate load spikes based on specific conditions."""
        if (
            self.sim_temperature > 42
            and self.sim_cpu_freq > 1.8
            and self.sim_network_usage > 400000
            and self.sim_error_rate < 0.02
        ):
            print("[EVENT] Load spike! (classic)")
            intensity = 1.0
            self.sim_temperature += random.uniform(4, 8) * intensity
            self.sim_disk_io += random.randint(int(1e5), int(5e5)) * intensity
            self.sim_error_rate = min(1.0, self.sim_error_rate + random.uniform(0.05, 0.15) * intensity)
            self.sim_network_usage += random.randint(int(1e5), int(5e5)) * intensity

        if (
            self.sim_battery > 95
            and self.sim_disk_io < 1e5
            and self.sim_cpu_freq < 1.3
        ):
            print("[EVENT] Load spike! (high battery, calm system)")
            intensity = 1.5
            self.sim_temperature += random.uniform(4, 8) * intensity
            self.sim_disk_io += random.randint(int(1e5), int(5e5)) * intensity
            self.sim_error_rate = min(1.0, self.sim_error_rate + random.uniform(0.05, 0.15) * intensity)
            self.sim_network_usage += random.randint(int(1e5), int(5e5)) * intensity

        if (
            self.sim_network_usage < 100000
            and 38 < self.sim_temperature < 45
        ):
            print("[EVENT] Load spike! (low net, moderate temp)")
            intensity = 0.7
            self.sim_temperature += random.uniform(4, 8) * intensity
            self.sim_disk_io += random.randint(int(1e5), int(5e5)) * intensity
            self.sim_error_rate = min(1.0, self.sim_error_rate + random.uniform(0.05, 0.15) * intensity)
            self.sim_network_usage += random.randint(int(1e5), int(5e5)) * intensity

    def apply_action(self, action_idx):
        """Apply the selected action to the simulated environment."""
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

        # Simulate load spikes after action
        self.load_spikes() 

        # Natural evolution
        self.sim_temperature += np.random.uniform(-0.2, 1.5) 
        self.sim_battery = max(0, self.sim_battery - np.random.uniform(0.3, 1.0))  
        self.sim_disk_io += np.random.randint(int(1e4), int(1e5))
        self.sim_error_rate = min(1.0, max(0, self.sim_error_rate + np.random.uniform(-0.01, 0.02)))
        self.sim_network_usage = max(0, self.sim_network_usage + np.random.randint(-int(1e4), int(5e4)))

    def compute_reward(self, state_before, state_after, action=None):
        """Hybrid reward: combines delta and absolute state penalties/bonuses."""
        delta_temp = state_after["temperature"] - state_before["temperature"]
        delta_battery = state_after["battery"] - state_before["battery"]
        delta_diskio = state_after["disk_write_bytes"] - state_before["disk_write_bytes"]
        delta_error = state_after["error_rate"] - state_before["error_rate"]
        delta_net = state_after["network_usage"] - state_before["network_usage"]

        # --- DELTA PART (variation-based, discriminant) ---
        reward = 0

        # TEMP
        if delta_temp > 1.5:
            reward -= 1.5 * (delta_temp ** 2)
        elif delta_temp < -1.0:
            reward += 1.2 * abs(delta_temp)

        # BATTERY
        if delta_battery < -0.6:
            reward -= 5 * abs(delta_battery)
        else:
            reward += 2 * delta_battery

        # ERROR RATE
        if delta_error > 0.01:
            reward -= 300 * delta_error
        elif delta_error < -0.005:
            reward += 100 * abs(delta_error)

        # NETWORK
        if delta_net > 20000 and delta_temp < 1 and delta_error < 0.005:
            reward += 0.002 * delta_net
        elif delta_net > 20000:
            reward -= 0.001 * delta_net

        # DISK I/O
        if delta_diskio < -100000 and delta_error > 0.005:
            reward -= 0.001 * abs(delta_diskio)
        elif delta_diskio < 0:
            reward += 0.0005 * abs(delta_diskio)

        # SYNERGY BONUS
        if delta_temp < 0 and delta_battery > 0 and delta_error < 0 and delta_net < 0:
            reward += 5

        # --- ABSOLUTE PART (state-based, safety/bonus) ---
        temp = state_after["temperature"]
        battery = state_after["battery"]
        diskio = state_after["disk_write_bytes"]
        error = state_after["error_rate"]
        net = state_after["network_usage"]

        # Strong penalty if temperature is critical
        if temp > 70:
            reward -= 50
        elif temp < 45:
            reward += 10

        # Battery thresholds
        if battery < 10:
            reward -= 20
        elif battery > 50:
            reward += 5

        # Disk I/O low bonus
        if abs(diskio) < 1e5:
            reward += 2

        # Error rate penalty
        if error > 0.2:
            reward -= 10

        # Penalize powersave if not needed
        if action is not None and action == "set_cpu_powersave":
            if state_after["temperature"] < 40 and state_after["cpu_freq"] < 1.2:
                reward -= 20 

        print(f"[REWARD] ΔTemp={delta_temp:.2f}, ΔBattery={delta_battery:.2f}, ΔDiskIO={delta_diskio:.0f}, ΔError={delta_error:.3f}, ΔNet={delta_net:.0f} | "
              f"Temp={temp:.1f}, Battery={battery:.1f}, Error={error:.3f} → Reward={reward:.2f}")
        return reward

    def save_q_table(self, path="q_table_iot.npy"):
        """Save the Q-table to a file."""
        np.save(path, self.q_table)

    def load_q_table(self, path="q_table_iot.npy"):
        """Load the Q-table from a file if it exists."""
        if os.path.exists(path):
            self.q_table = np.load(path)
            print("[Q-TABLE] Loaded from file.")
