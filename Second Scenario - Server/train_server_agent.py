import time
import psutil
import os
from agent_server import ServerAgent
from load_generator import run_wrk
import numpy as np
import matplotlib.pyplot as plt

def collect_metrics(requests_per_sec):
    """
    Collect system metrics and return them as a dictionary.
    """
    cpu = psutil.cpu_percent(interval=0.1)
    iowait = psutil.cpu_times_percent(interval=0.1).iowait
    interrupts = psutil.cpu_stats().interrupts
    ctx_switches = psutil.cpu_stats().ctx_switches
    net = psutil.net_io_counters()
    net_usage = (net.bytes_sent + net.bytes_recv) / 1e6  # MB
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

def train_agent(num_episodes=500, nb_steps_per_episode=10):
    """
    Train the ServerAgent using Q-learning over multiple episodes.
    """
    agent = ServerAgent()
    qtable_path = "Second Scenario - Server/q_table_server.npy"
    if os.path.exists(qtable_path):
        agent.load_q_table(qtable_path)
    rewards = []

    for episode in range(num_episodes):
        reset_sys_params()
        requests_per_sec, _ = run_wrk(duration=2)  
        state = agent.get_state(collect_metrics(requests_per_sec))
        total_reward = 0

        for step in range(nb_steps_per_episode):
            action_idx = agent.select_action(state)
            agent.apply_action(action_idx)
            
            requests_per_sec, _ = run_wrk(duration=2)
            next_state = agent.get_state(collect_metrics(requests_per_sec))
            
            reward = agent.compute_reward(collect_metrics(requests_per_sec))
            
            agent.learn(state, action_idx, reward, next_state)
            state = next_state
            total_reward += reward

        requests_per_sec, _ = run_wrk(duration=5)
        metrics = collect_metrics(requests_per_sec)
        reward = agent.compute_reward(metrics, debug=True)
        agent.learn(state, action_idx, reward, state)
        rewards.append(reward)
        agent.exploration_rate = max(0.05, agent.exploration_rate * agent.exploration_decay)
        time.sleep(1)

    agent.save_q_table(qtable_path)

    window = min(10, len(rewards))
    if len(rewards) >= 2:
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        plt.figure(figsize=(10,5))
        plt.plot(range(window, window + len(moving_avg)), moving_avg, label=f"Moving average ({window})")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("Moving average of reward")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    train_agent()