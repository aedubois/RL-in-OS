import matplotlib.pyplot as plt
import noop_policy_iot
import random_agent_iot
import heuristic_agent_iot
import train_iot_agent
import numpy as np

NUM_EPISODES = 1000
SLEEP_INTERVAL = 0.0001

window = 10

def moving_average(data, window):
    """Calculate the moving average of a given data array."""
    return np.convolve(data, np.ones(window)/window, mode='valid')

def run_and_get_rewards(agent_main_func, rewards_var_name):
    """Run the agent's main function and return the rewards."""
    rewards = agent_main_func(
        num_episodes=NUM_EPISODES,
        sleep_interval=SLEEP_INTERVAL,
        return_rewards=True  
    )
    return np.array(rewards)

def main():
    """Main function to compare strategies in the IoT scenario."""
    print("Running no-op baseline policy...")
    rewards_noop = run_and_get_rewards(noop_policy_iot.main, "rewards")
    print("Running random policy...")
    rewards_random = run_and_get_rewards(random_agent_iot.main, "rewards")
    print("Running heuristic policy...")
    rewards_heuristic = run_and_get_rewards(heuristic_agent_iot.main, "rewards")
    print("Running RL policy...")
    rewards_rl = run_and_get_rewards(train_iot_agent.main, "rewards")

    plt.figure(figsize=(12, 6))
#    plt.plot(rewards_random, label=f"Random (mean={np.mean(rewards_random):.1f})")
#    plt.plot(rewards_heuristic, label=f"Heuristic (mean={np.mean(rewards_heuristic):.1f})")
#    plt.plot(rewards_rl, label=f"RL (mean={np.mean(rewards_rl):.1f})")
#    plt.plot(rewards_noop, label=f"No-op (mean={np.mean(rewards_noop):.1f})")

    window = 15
    for rewards, label, style in [
        (rewards_random, "Random", "blue"),
        (rewards_heuristic, "Heuristic", "orange"),
        (rewards_rl, "RL", "green"),
        (rewards_noop, "No-op", "red"),
    ]:
        if len(rewards) >= window:
            moving_avg = np.convolve(rewards, np.ones(window)/window, mode="valid")
            plt.plot(range(window, window + len(moving_avg)), moving_avg, style, alpha=0.6, label=f"{label} (moving avg)")

    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("IoT Scenario: Reward per Episode for Each Policy")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("Third Scenario - IoT/compare_strategies_iot.png")
    plt.show()

    print("\n=== Reward Statistics ===")
    print(f"Random: mean = {np.mean(rewards_random):.2f}, std = {np.std(rewards_random):.2f}")
    print(f"Heuristic: mean = {np.mean(rewards_heuristic):.2f}, std = {np.std(rewards_heuristic):.2f}")
    print(f"RL: mean = {np.mean(rewards_rl):.2f}, std = {np.std(rewards_rl):.2f}")
    print(f"No-op: mean = {np.mean(rewards_noop):.2f}, std = {np.std(rewards_noop):.2f}")

if __name__ == "__main__":
    main()