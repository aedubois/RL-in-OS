import matplotlib.pyplot as plt
import random_agent_iot
import heuristic_agent_iot
import train_iot_agent
import numpy as np

NUM_EPISODES = 100
SLEEP_INTERVAL = 0.01

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
    print("Running random policy...")
    rewards_random = run_and_get_rewards(random_agent_iot.main, "rewards")
    print("Running heuristic policy...")
    rewards_heuristic = run_and_get_rewards(heuristic_agent_iot.main, "rewards")
    print("Running RL policy...")
    rewards_rl = run_and_get_rewards(train_iot_agent.main, "rewards")

    plt.figure(figsize=(10, 6))
    plt.plot(rewards_random, label=f"Random (mean={np.mean(rewards_random):.1f})")
    plt.plot(rewards_heuristic, label=f"Heuristic (mean={np.mean(rewards_heuristic):.1f})")
    plt.plot(rewards_rl, label=f"RL (mean={np.mean(rewards_rl):.1f})")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("IoT Scenario: Reward per Episode for Each Policy")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("compare_strategies_iot.png")
    plt.show()

    print("\n=== Reward Statistics ===")
    print(f"Random: mean = {np.mean(rewards_random):.2f}, std = {np.std(rewards_random):.2f}")
    print(f"Heuristic: mean = {np.mean(rewards_heuristic):.2f}, std = {np.std(rewards_heuristic):.2f}")
    print(f"RL: mean = {np.mean(rewards_rl):.2f}, std = {np.std(rewards_rl):.2f}")

if __name__ == "__main__":
    main()