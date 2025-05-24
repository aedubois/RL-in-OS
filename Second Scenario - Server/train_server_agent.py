import time
import psutil
import os
from agent_server import ServerAgent
from load_generator import run_wrk
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict
import json
from datetime import datetime

@dataclass
class Configuration:
    params: Dict
    reward: float
    rps: float
    latency: float
    timestamp: str

    def to_dict(self):
        return {
            "params": self.params,
            "reward": self.reward,
            "rps": self.rps,
            "latency": self.latency,
            "timestamp": self.timestamp
        }

def collect_metrics(requests_per_sec):
    """
    Collect system metrics and return them as a dictionary.
    """
    cpu = psutil.cpu_percent(interval=0.1)
    iowait = psutil.cpu_times_percent(interval=0.1).iowait
    interrupts = psutil.cpu_stats().interrupts
    ctx_switches = psutil.cpu_stats().ctx_switches
    net = psutil.net_io_counters()
    net_usage = (net.bytes_sent + net.bytes_recv) / 1e6
    return {
        "cpu_usage": cpu,
        "iowait": iowait,
        "interrupts": interrupts,
        "ctx_switches": ctx_switches,
        "net_usage": net_usage,
        "requests_per_sec": requests_per_sec
    }

def reset_sys_params():
    """
    Reset system parameters to default values between episodes.
    """
    os.system("sudo sysctl -w vm.dirty_ratio=20")
    os.system("sudo sysctl -w net.core.rmem_max=212992")
    os.system("sudo sysctl -w net.core.wmem_max=212992")
    os.system("sudo sh -c 'echo 1 > /sys/module/zswap/parameters/enabled'")
    os.system("sudo renice 0 -p $(pgrep nginx)")
    os.system("sudo truncate -s 0 /var/log/nginx/access.log")

def get_current_params():
    dirty_ratio = os.popen("sysctl vm.dirty_ratio").read().split("=")[1].strip()
    rmem = os.popen("sysctl net.core.rmem_max").read().split("=")[1].strip()
    wmem = os.popen("sysctl net.core.wmem_max").read().split("=")[1].strip()
    zswap = os.popen("cat /sys/module/zswap/parameters/enabled").read().strip()
    return {
        "dirty_ratio": dirty_ratio,
        "rmem_max": rmem,
        "wmem_max": wmem,
        "zswap": zswap
    }

def validate_configuration(config_params, agent):
    for param, value in config_params.items():
        if param == "dirty_ratio":
            os.system(f"sudo sysctl -w vm.dirty_ratio={value}")
        elif param == "rmem_max":
            os.system(f"sudo sysctl -w net.core.rmem_max={value}")
        elif param == "wmem_max":
            os.system(f"sudo sysctl -w net.core.wmem_max={value}")
        elif param == "zswap":
            os.system(f"sudo sh -c 'echo {value} > /sys/module/zswap/parameters/enabled'")
    rps, latency, p99, _ = run_wrk(duration=10)
    metrics = collect_metrics(rps)
    reward = agent.compute_reward(metrics, latency=latency, p99=p99)
    return reward, rps, latency

def train_agent(num_episodes=100, nb_steps_per_episode=10):
    """
    Train the ServerAgent using Q-learning over multiple episodes.
    """
    agent = ServerAgent()
    qtable_path = "Second Scenario - Server/q_table_server.npy"
    if os.path.exists(qtable_path):
        agent.load_q_table(qtable_path)
    rewards = []
    previous_actions = []
    best_configs = []
    best_reward = float('-inf')

    try:
        for episode in range(num_episodes):
            print(f"\n=== Episode {episode+1} / {num_episodes} ===")
            reset_sys_params()
            requests_per_sec, latency, p99, _ = run_wrk(duration=2)
            state = agent.get_state(collect_metrics(requests_per_sec))
            total_reward = 0

            for step in range(nb_steps_per_episode):
                action_idx = agent.select_action(state)
                max_attempts = 3
                for _ in range(max_attempts):
                    if agent.apply_action(action_idx):
                        break
                    action_idx = agent.select_action(state)
                else:
                    continue

                requests_per_sec, latency, p99, _ = run_wrk(duration=2)
                next_state = agent.get_state(collect_metrics(requests_per_sec))
                metrics = collect_metrics(requests_per_sec)
                reward = agent.compute_reward(metrics, latency=latency, p99=p99)
            
                penalty_factor = agent.penalize_consecutive_actions(action_idx, previous_actions)
                reward *= penalty_factor
                previous_actions.append(action_idx)
                agent.learn(state, action_idx, reward, next_state)
                state = next_state
                total_reward += reward

            requests_per_sec, latency, p99, _ = run_wrk(duration=5)
            metrics = collect_metrics(requests_per_sec)
            reward = agent.compute_reward(metrics, latency=latency, p99=p99)
            agent.learn(state, action_idx, reward, state)
            rewards.append(reward)

            print(f"Reward of episode {episode+1} : {reward}") 

            if reward > best_reward:
                best_reward = reward
                config = Configuration(
                    params=get_current_params(),
                    reward=reward,
                    rps=requests_per_sec,
                    latency=latency,
                    timestamp=datetime.now().isoformat()
                )
                best_configs.append(config)
                best_configs = sorted(best_configs, key=lambda x: x.reward, reverse=True)[:5]
                with open('Second Scenario - Server/best_configs.json', 'w') as f:
                    json.dump([c.to_dict() for c in best_configs], f, indent=2)

            agent.exploration_rate = max(0.05, agent.exploration_rate * agent.exploration_decay)
            time.sleep(1)

        print("List of rewards:", rewards)
        agent.save_q_table(qtable_path)

        plots_dir = "Second Scenario - Server/plots"
        os.makedirs(plots_dir, exist_ok=True)

        existing = [int(f.split("_")[1].split(".")[0]) for f in os.listdir(plots_dir) if f.startswith("plot_") and f.endswith(".png")]
        next_num = max(existing) + 1 if existing else 1
        plot_path = os.path.join(plots_dir, f"plot_{next_num}.png")

        window = min(10, len(rewards))  # Défini AVANT le test
        plt.figure(figsize=(10,5))
        plt.plot(range(1, len(rewards)+1), rewards, label="Reward per episode", alpha=0.7)
        if len(rewards) >= window:
            moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
            plt.plot(range(window, window + len(moving_avg)), moving_avg, label=f"Moving average ({window})", linewidth=2)
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("Reward per episode and moving average")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.show()
        print(f"Plot saved as {plot_path}")

        print("\nBest configurations validation:")
        for config in best_configs:
            print(f"\nTesting configuration: {config.params}")
            reward, rps, latency = validate_configuration(config.params, agent)
            print(f"Validation - RPS: {rps:.2f}, Latency: {latency:.2f}ms, Reward: {reward:.2f}")
    except KeyboardInterrupt:
        print("\nKeyboard interrupt detected. Saving Q-table and cleaning up...")
        agent.save_q_table(qtable_path)
        reset_sys_params()
        print("Q-table saved. System parameters reset. Exiting.")

if __name__ == "__main__":
    train_agent()