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
    """Data class to hold configuration parameters and their associated metrics."""
    params: Dict
    reward: float
    rps: float
    latency: float
    timestamp: str

    def to_dict(self):
        """Convert the configuration to a dictionary for JSON serialization."""
        return {
            "params": self.params,
            "reward": self.reward,
            "rps": self.rps,
            "latency": self.latency,
            "timestamp": self.timestamp
        }

def collect_metrics(requests_per_sec, latency):
    """Collect nginx-specific metrics and return them as a dictionary"""
    import psutil
    nginx_pids = [p.pid for p in psutil.process_iter(['name']) if p.info['name'] == 'nginx']
    cpu = sum(psutil.Process(pid).cpu_percent(interval=0.1) for pid in nginx_pids) if nginx_pids else 0.0
    mem = sum(psutil.Process(pid).memory_info().rss for pid in nginx_pids) / 1e6 if nginx_pids else 0.0
    return {
        "cpu_usage": cpu,
        "mem_usage": mem,
        "requests_per_sec": requests_per_sec,
        "latency": latency if latency is not None else 0.0
    }

def reset_sys_params():
    """Reset system parameters to default values between episodes."""
    os.system("sudo sysctl -w vm.dirty_ratio=20")
    os.system("sudo sysctl -w net.core.rmem_max=212992")
    os.system("sudo sysctl -w net.core.wmem_max=212992")
    os.system("sudo sysctl -w net.ipv4.tcp_tw_reuse=0")
    os.system("sudo sysctl -w net.ipv4.tcp_fin_timeout=60") 
    os.system("sudo sysctl -w net.core.somaxconn=128")
    os.system("sudo systemctl restart nginx")
    os.system("sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'")
    os.system("pkill wrk")
    os.system("sudo truncate -s 0 /var/log/nginx/access.log")
    time.sleep(2)  

def get_current_params():
    """Get current system parameters for logging and validation."""
    params = {}
    params["dirty_ratio"] = os.popen("sysctl vm.dirty_ratio").read().split("=")[1].strip()
    params["rmem_max"] = os.popen("sysctl net.core.rmem_max").read().split("=")[1].strip()
    params["wmem_max"] = os.popen("sysctl net.core.wmem_max").read().split("=")[1].strip()
    params["tcp_tw_reuse"] = os.popen("sysctl net.ipv4.tcp_tw_reuse").read().split("=")[1].strip()
    params["tcp_fin_timeout"] = os.popen("sysctl net.ipv4.tcp_fin_timeout").read().split("=")[1].strip()
    params["somaxconn"] = os.popen("sysctl net.core.somaxconn").read().split("=")[1].strip()
    return params

def validate_configuration(config_params, agent):
    """Validate a specific configuration by applying it and running a load test."""
    rps, latency, p99, _ = run_wrk(duration=10)
    metrics = collect_metrics(rps, latency)  
    reward = agent.compute_reward(metrics, latency=latency, p99=p99)
    return reward, rps, latency

def run_episode(agent, nb_steps_per_episode, sleep_interval, previous_actions):
    """Run a single episode of the reinforcement learning agent on the server environment."""
    reset_sys_params()
    requests_per_sec, latency, p99, _ = run_wrk(duration=2)
    print(f"wrk RPS: {requests_per_sec}, latency: {latency} ms, p99: {p99} ms")
    state = agent.get_state(collect_metrics(requests_per_sec, latency))
    total_reward = 0
    last_rps = requests_per_sec

    for step in range(nb_steps_per_episode):
        action_idx = agent.select_action(state)
        print(f"Applying action: {agent.actions[action_idx]}")
        agent.apply_action(action_idx)
        requests_per_sec, latency, p99, _ = run_wrk(duration=2)
        next_state = agent.get_state(collect_metrics(requests_per_sec, latency))
        metrics = collect_metrics(requests_per_sec, latency)
        print("metrics:", metrics)
        reward = agent.compute_reward(metrics, latency=latency, p99=p99, prev_rps=last_rps)
        last_rps = requests_per_sec
        if agent.actions[action_idx] != "no_op":
            penalty_factor = agent.penalize_consecutive_actions(action_idx, previous_actions)
            reward *= penalty_factor
        print("reward:", reward)
        previous_actions.append(action_idx)
        agent.learn(state, action_idx, reward, next_state)
        state = next_state
        total_reward += reward 
        time.sleep(sleep_interval)

    # Final evaluation step 
    requests_per_sec, latency, p99, _ = run_wrk(duration=5)
    metrics = collect_metrics(requests_per_sec, latency)
    reward = agent.compute_reward(metrics, latency=latency, p99=p99)
    agent.learn(state, action_idx, reward, state)
    return total_reward, requests_per_sec, latency 

def update_best_configs(best_configs, reward, requests_per_sec, latency):
    """Update the list of best configurations with the current episode's results."""
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
    return best_configs

def plot_rewards(rewards, plots_dir):
    """ Plot the rewards and save the plot to a file."""
    os.makedirs(plots_dir, exist_ok=True)
    existing = [int(f.split("_")[1].split(".")[0]) for f in os.listdir(plots_dir) if f.startswith("plot_") and f.endswith(".png")]
    next_num = max(existing) + 1 if existing else 1
    plot_path = os.path.join(plots_dir, f"plot_{next_num}.png")
    window = min(10, len(rewards))
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
    print(f"Plot saved as {plot_path}")

def train_agent(num_episodes=100, nb_steps_per_episode=20, sleep_interval=0.1, return_rewards=False, exploration_rate=1.0):
    """Train a reinforcement learning agent for the server scenario."""
    agent = ServerAgent(exploration_rate=exploration_rate)
    qtable_path = "Second Scenario - Server/q_table_server.npy"
    rewards_dir = "Second Scenario - Server/rewards"
    os.makedirs(rewards_dir, exist_ok=True)
    rewards_path = os.path.join(rewards_dir, "rewards_rl_server.npy")
    if os.path.exists(qtable_path):
        agent.load_q_table(qtable_path)
    rewards = []
    previous_actions = []
    best_configs = []
    best_reward = float('-inf')

    try:
        for episode in range(num_episodes):
            print(f"\n=== Episode {episode+1} / {num_episodes} ===")
            reward, requests_per_sec, latency = run_episode(agent, nb_steps_per_episode, sleep_interval, previous_actions)
            rewards.append(reward/nb_steps_per_episode)
            print(f"Average reward of episode {episode+1} : {reward/nb_steps_per_episode}")

            if reward > best_reward:
                best_reward = reward
                best_configs = update_best_configs(best_configs, reward, requests_per_sec, latency)

            agent.exploration_rate = max(0.05, agent.exploration_rate * agent.exploration_decay)

        agent.save_q_table(qtable_path)
        np.save(rewards_path, np.array(rewards))
        plots_dir = "Second Scenario - Server/plots"
        plot_rewards(rewards, plots_dir)

        print("\nBest configurations validation:")
        for config in best_configs:
            print(f"\nTesting configuration: {config.params}")
            reward, rps, latency = validate_configuration(config.params, agent)
            print(f"Validation - RPS: {rps:.2f}, Latency: {latency:.2f}ms, Reward: {reward:.2f}")
        if return_rewards:
            return rewards

    except KeyboardInterrupt:
        print("\nKeyboard interrupt detected. Saving Q-table and cleaning up...")
        agent.save_q_table(qtable_path)
        reset_sys_params()
        print("Q-table saved. System parameters reset. Exiting.")

if __name__ == "__main__":
    train_agent()