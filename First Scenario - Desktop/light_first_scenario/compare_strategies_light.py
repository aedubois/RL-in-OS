import matplotlib.pyplot as plt
import numpy as np
from light_noop_policy import noop_policy
from light_random_policy import random_policy
from light_heuristic_policy import heuristic_policy
from light_train_agent import train_agent

NUM_EPISODES = 50
NB_STEPS_PER_EPISODE = 10
SLEEP_INTERVAL = 2
WINDOW = 10

def moving_average(data, window):
    """Calculate the moving average of a given data array."""
    return np.convolve(data, np.ones(window)/window, mode='valid')

def run_and_get_rewards(agent_func, **kwargs):
    """Run the agent's function and return the rewards as a numpy array."""
    rewards = agent_func(
        num_episodes=NUM_EPISODES,
        nb_steps_per_episode=NB_STEPS_PER_EPISODE,
        **kwargs
    )
    return np.array(rewards)

def main():
    print("Running Random policy...")
    rewards_random = run_and_get_rewards(random_policy)
    print("Running No-op policy...")
    rewards_noop = run_and_get_rewards(noop_policy)
    print("Running Heuristic policy...")
    rewards_heuristic = run_and_get_rewards(heuristic_policy)
    print("Running RL agent (full exploitation)...")
    rewards_rl = run_and_get_rewards(
        train_agent,
        exploration_rate=0.0,
        exploration_decay=1.0,
        learning_rate=0.1,
        discount_factor=0.9
    )

    plt.figure(figsize=(12, 6))
    for rewards, label, color in [
        (rewards_random, "Random", "blue"),
        (rewards_heuristic, "Heuristic", "orange"),
        (rewards_rl, "RL (exploitation)", "green"),
        (rewards_noop, "No-op", "red"),
    ]:
        if len(rewards) >= WINDOW:
            moving_avg = moving_average(rewards, WINDOW)
            plt.plot(range(WINDOW, WINDOW + len(moving_avg)), moving_avg, color=color, alpha=0.7, label=f"{label} (moving avg)")

    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Light Scenario: Moving Average Reward per Episode for Each Policy")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("First Scenario - Desktop/light_first_scenario/compare_strategies_light.png")
    plt.show()

    print("\n=== Reward Statistics ===")
    print(f"Random: mean = {np.mean(rewards_random):.2f}, std = {np.std(rewards_random):.2f}")
    print(f"Heuristic: mean = {np.mean(rewards_heuristic):.2f}, std = {np.std(rewards_heuristic):.2f}")
    print(f"RL: mean = {np.mean(rewards_rl):.2f}, std = {np.std(rewards_rl):.2f}")
    print(f"No-op: mean = {np.mean(rewards_noop):.2f}, std = {np.std(rewards_noop):.2f}")

if __name__ == "__main__":
    main()