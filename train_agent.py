import time
import numpy as np
import random
import os
import subprocess
from agent import EventAgent
import matplotlib.pyplot as plt
import pandas as pd

NEGATIVE_ACTIONS = [
    "simulate_cpu_stress",
    "simulate_memory_stress",
    "simulate_disk_fill",
    "simulate_disk_latency",
    "stress_tmpfs",
    "play_streaming_video",
    "simulate_network_stress",
    "simulate_swap_stress",
    "simulate_high_load",
    "simulate_temp_increase"
]

def apply_negative_action(action):
    """
    Launches a negative action (system stress) based on the action name.
    """
    if action == "simulate_cpu_stress":
        return subprocess.Popen("stress-ng --cpu 2 --timeout 8", shell=True)
    elif action == "simulate_memory_stress":
        return subprocess.Popen("stress-ng --vm 2 --vm-bytes 1G --timeout 8", shell=True)
    elif action == "simulate_disk_fill":
        return subprocess.Popen("dd if=/dev/zero of=/tmp/fillfile bs=1M count=2048", shell=True)
    elif action == "simulate_disk_latency":
        return subprocess.Popen("stress-ng --hdd 2 --hdd-bytes 1G --timeout 8", shell=True)
    elif action == "stress_tmpfs":
        return subprocess.Popen("dd if=/dev/zero of=/dev/shm/tmpfs_stress bs=1M count=1024", shell=True)
    elif action == "play_streaming_video":
        return subprocess.Popen("vlc --intf dummy --run-time=8 --play-and-exit https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4 vlc://quit", shell=True)
    elif action == "simulate_network_stress":
        server_proc = subprocess.Popen("iperf3 -s", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(1)
        client_proc = subprocess.Popen("iperf3 -c 127.0.0.1 -t 8", shell=True)
        return (server_proc, client_proc)
    elif action == "simulate_swap_stress":
        return subprocess.Popen("stress-ng --swap 2 --timeout 8", shell=True)
    elif action == "simulate_high_load":
        return subprocess.Popen("timeout 8s yes > /dev/null", shell=True)
    elif action == "simulate_temp_increase":
        return subprocess.Popen("stress-ng --cpu 4 --timeout 8", shell=True)
    else:
        return None

def clean_resources():
    """
    Cleans up any leftover processes or files from stress actions.
    """
    os.system("pkill -f stress-ng")
    os.system("pkill -f yes")
    os.system("pkill -f vlc")
    os.system("pkill -f iperf3")
    os.system("rm -f /tmp/fillfile")
    os.system("rm -f /dev/shm/tmpfs_stress")

def train_agent(num_episodes=100, learning_rate=0.1, discount_factor=0.9, exploration_rate=1.0, exploration_decay=0.995):
    """
    Main training loop for the RL agent.
    """
    agent = EventAgent()
    rewards_per_episode = []

    for episode in range(num_episodes):
        negative_action = random.choice(NEGATIVE_ACTIONS)
        proc = apply_negative_action(negative_action)
        time.sleep(4)

        agent.update_metrics_once()
        state = agent.get_normalized_state()

        if negative_action == "simulate_network_stress" and proc is not None:
            server_proc, client_proc = proc
            client_proc.wait()
            server_proc.terminate()
            server_proc.wait()
        else:
            if proc is not None:
                proc.wait()

        if random.uniform(0, 1) < exploration_rate:
            action_idx = random.randint(0, len(agent.actions) - 1)
        else:
            action_idx = agent.select_action(state)
        agent.apply_action(action_idx)
        time.sleep(4)

        agent.update_metrics_once()
        new_state = agent.get_normalized_state()

        reward = agent.compute_reward(state, new_state, debug=True)

        state_idx = agent.get_discretized_state()
        new_state_idx = agent.get_discretized_state()
        agent.q_table[state_idx][action_idx] = agent.q_table[state_idx][action_idx] + learning_rate * (
            reward + discount_factor * np.max(agent.q_table[new_state_idx]) - agent.q_table[state_idx][action_idx]
        )

        exploration_rate = max(0.05, exploration_rate * exploration_decay)
        rewards_per_episode.append(reward)

        clean_resources()

    np.save("q_table.npy", agent.q_table)

    pd.Series(rewards_per_episode).rolling(10).mean().plot(title="Mean Reward (window=10)")
    plt.show()

if __name__ == "__main__":
    train_agent()
