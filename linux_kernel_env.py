import gymnasium as gym
from gymnasium import spaces
import numpy as np

class KernelTuneEnv(gym.Env):
    """
    Environnement pour ajuster les paramètres du noyau Linux et observer les effets sur les métriques système.
    """
    def __init__(self):
        super(KernelTuneEnv, self).__init__()
        self.action_space = spaces.Discrete(6)  # 6 actions possibles
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(6,), dtype=np.float32)
        self.state = None
        self.step_count = 0
        self.max_steps = 1000

    def reset(self, seed=None, options=None):
        """
        Réinitialise l'environnement à son état initial.
        """
        super().reset(seed=seed)
        self.state = self._get_initial_state()
        self.step_count = 0
        return self.state, {}

    def step(self, action):
        """
        Applique une action et retourne l'état suivant, la récompense, et les indicateurs de fin d'épisode.
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
        Génère l'état initial de l'environnement.
        """
        return np.random.uniform(low=0.0, high=1.0, size=(6,)).astype(np.float32)

    def _apply_action(self, action):
        """
        Applique une action pour influencer les métriques.
        """
        if action == 0:  # Réduire l'utilisation CPU
            self.state[0] = max(0.0, self.state[0] - 0.1)
        elif action == 1:  # Réduire l'utilisation RAM
            self.state[3] = max(0.0, self.state[3] - 0.1)
            self.state[5] = min(1.0, self.state[5] + 0.05)  # Augmente légèrement la latence
        elif action == 2:  # Réduire l'attente I/O
            self.state[4] = max(0.0, self.state[4] - 0.05)
        elif action == 3:  # Réduire la latence
            self.state[5] = max(0.0, self.state[5] - 0.1)
        elif action == 4:  # Réduire le Swap (burst rare)
            self.state[1] = max(0.0, self.state[1] - 0.2)
        elif action == 5:  # Réduire la charge système (Load avg)
            self.state[2] = max(0.0, self.state[2] - 0.1)

    def _get_next_state(self):
        """
        Génère le prochain état en ajoutant un bruit aléatoire.
        """
        noise = np.random.normal(loc=0.0, scale=0.02, size=(6,))
        next_state = self.state + noise
        return np.clip(next_state, 0.0, 1.0).astype(np.float32)

    def _calculate_reward(self):
        """
        Calcule la récompense en fonction des objectifs définis.
        """
        # Contraintes
        cpu_target = 0.6  # 60%
        iowait_target = 0.05  # 5%
        ram_target = 0.7  # 70%
        swap_target = 0.0  # Proche de 0
        latency_target = 0.1  # 100ms
        load_target = 0.5  # Charge relative au nombre de cœurs (normalisée)

        # Pénalités si les contraintes ne sont pas respectées
        cpu_penalty = max(0, self.state[0] - cpu_target)
        iowait_penalty = max(0, self.state[4] - iowait_target)
        ram_penalty = max(0, self.state[3] - ram_target)
        swap_penalty = max(0, self.state[1] - swap_target)
        latency_penalty = max(0, self.state[5] - latency_target)
        load_penalty = max(0, self.state[2] - load_target)

        # Calcul de la récompense
        reward = 1.0 - (
            0.3 * cpu_penalty +
            0.2 * iowait_penalty +
            0.2 * ram_penalty +
            0.1 * swap_penalty +
            0.1 * latency_penalty +
            0.1 * load_penalty
        )
        return max(reward, 0.0)  # La récompense ne peut pas être négative

    def render(self):
        """
        Affiche l'état actuel de l'environnement.
        """
        print(f"État actuel : {self.state}")

    def close(self):
        """
        Nettoie les ressources utilisées par l'environnement.
        """
        pass
