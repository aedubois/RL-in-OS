import numpy as np
import matplotlib.pyplot as plt
import os

REWARDS_DIR = "Second Scenario - Server/rewards"

def load_rewards(filename):
    path = os.path.join(REWARDS_DIR, filename)
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return None
    return np.load(path)

def moving_average(data, window=5):
    return np.convolve(data, np.ones(window)/window, mode='valid')

def main():
    rewards_files = {
        "No-op": "rewards_noop_server.npy",
        "Random": "rewards_random_server.npy",
        "Heuristic": "rewards_heuristic_server.npy",
        "RL": "rewards_rl_server.npy"
    }

    rewards_data = {}
    for label, fname in rewards_files.items():
        data = load_rewards(fname)
        if data is not None:
            rewards_data[label] = data

    plt.figure(figsize=(12, 6))
    window = 5
    colors = {
        "No-op": "red",
        "Random": "blue",
        "Heuristic": "orange",
        "RL": "green"
    }

    for label, data in rewards_data.items():
        plt.plot(
            moving_average(data, window),
            color=colors[label],
            label=f"{label} (moving avg)"
        )

    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Server Scenario: Reward per Episode for Each Policy")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("Second Scenario - Server/compare_strategies_server.png")
    plt.show()

    print("\n=== Reward Statistics ===")
    for label, data in rewards_data.items():
        print(f"{label}: mean = {np.mean(data):.2f}, std = {np.std(data):.2f}")

if __name__ == "__main__":
    main()