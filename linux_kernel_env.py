import gymnasium as gym
from gymnasium import spaces
import numpy as np

class KernelTuneEnv(gym.Env):
    """
    Environment for tuning Linux kernel parameters and observing the effects on system metrics.
    """
    def __init__(self):
        super(KernelTuneEnv, self).__init__()
        self.action_space = spaces.Discrete(6)  # 6 possible actions
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(6,), dtype=np.float32)
        self.state = None
        self.step_count = 0
        self.max_steps = 1000

    def reset(self, seed=None, options=None):
        """
        Resets the environment to its initial state.
        """
        super().reset(seed=seed)
        self.state = self._get_initial_state()
        self.step_count = 0
        return self.state, {}

    def step(self, action):
        """
        Applies an action and returns the next state, reward, and episode termination indicators.
        """
        self._apply_action(action)
        self.state = self._get_next_state()
        reward = self._calculate_reward()
        self.step_count += 1
        terminated = self.step_count >= self.max_steps
        truncated = False
        return self.state, reward, terminated, truncated, {}

    def _get_initial_state(self):
        """
        Generates the initial state of the environment.
        """
        return np.random.uniform(low=0.0, high=1.0, size=(6,)).astype(np.float32)

    def _apply_action(self, action):
        """
        Applies an action to influence the metrics.
        """
        if action == 0:  # Reduce CPU usage
            self.state[0] = max(0.0, self.state[0] - 0.1)
        elif action == 1:  # Reduce RAM usage
            self.state[3] = max(0.0, self.state[3] - 0.1)
            self.state[5] = min(1.0, self.state[5] + 0.05)  # Slightly increase latency
        elif action == 2:  # Reduce I/O wait
            self.state[4] = max(0.0, self.state[4] - 0.05)
        elif action == 3:  # Reduce latency
            self.state[5] = max(0.0, self.state[5] - 0.1)
        elif action == 4:  # Reduce Swap (rare bursts)
            self.state[1] = max(0.0, self.state[1] - 0.2)
        elif action == 5:  # Reduce system load (Load avg)
            self.state[2] = max(0.0, self.state[2] - 0.1)

    def _get_next_state(self):
        """
        Generates the next state by adding random noise.
        """
        noise = np.random.normal(loc=0.0, scale=0.02, size=(6,))
        next_state = self.state + noise
        return np.clip(next_state, 0.0, 1.0).astype(np.float32)

    def _calculate_reward(self):
        """
        Calculates the reward based on defined objectives.
        """
        # Targets
        cpu_target = 0.6  # 60%
        iowait_target = 0.05  # 5%
        ram_target = 0.7  # 70%
        swap_target = 0.0  # Close to 0
        latency_target = 0.1  # 100ms
        load_target = 0.5  # Load relative to the number of cores (normalized)

        # Penalties if targets are not met
        cpu_penalty = max(0, self.state[0] - cpu_target)
        iowait_penalty = max(0, self.state[4] - iowait_target)
        ram_penalty = max(0, self.state[3] - ram_target)
        swap_penalty = max(0, self.state[1] - swap_target)
        latency_penalty = max(0, self.state[5] - latency_target)
        load_penalty = max(0, self.state[2] - load_target)

        # Reward calculation
        reward = 1.0 - (
            0.3 * cpu_penalty +
            0.2 * iowait_penalty +
            0.2 * ram_penalty +
            0.1 * swap_penalty +
            0.1 * latency_penalty +
            0.1 * load_penalty
        )
        return max(reward, 0.0)  # Reward cannot be negative

    def render(self):
        """
        Displays the current state of the environment.
        """
        print(f"Current state: {self.state}")

    def close(self):
        """
        Cleans up resources used by the environment.
        """
        pass
