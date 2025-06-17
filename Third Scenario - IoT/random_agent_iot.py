import time
import numpy as np
from agent_iot import IoTAgent
import matplotlib.pyplot as plt
import os

def main(num_episodes=100, sleep_interval=0.1, return_rewards=False):
    """Run a random policy for the IoT agent."""
    agent = IoTAgent()
    rewards = []

    for episode in range(num_episodes):
        print(f"\n=== Episode {episode + 1} / {num_episodes} ===")
        state = agent.reset()
        episode_reward = 0

        for step in range(100):
            state_tuple = agent.normalize_state(state)
            action_idx = np.random.randint(len(agent.actions))
            agent.apply_action(action_idx)
            next_state = agent.get_state()
            reward = agent.compute_reward(state, next_state)
            episode_reward += reward
            state = next_state

            if state["battery"] < 5 or state["error_rate"] > 0.8:
                print("[INFO] Battery too low or error rate too high, ending episode.")
                break

            time.sleep(sleep_interval)

        rewards.append(episode_reward)

    if return_rewards:
        return rewards

    np.save("rewards_random_iot.npy", np.array(rewards))
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Random Policy - IoT")
    plt.savefig("random_policy_iot.png")
    plt.show()

if __name__ == "__main__":
    main(num_episodes=100, sleep_interval=0.1)