import time
import numpy as np
from agent_iot import IoTAgent
import matplotlib.pyplot as plt

def main(num_episodes=100, sleep_interval=0.1, return_rewards=False):
    """Run a no-op policy for the IoT agent."""
    agent = IoTAgent()
    rewards = []

    for episode in range(num_episodes):
        state = agent.reset()
        episode_reward = 0

        for step in range(100):
            action_idx = agent.actions.index("no_op")
            agent.apply_action(action_idx)
            next_state = agent.get_state()
            reward = agent.compute_reward(state, next_state, action_idx)
            episode_reward += reward
            state = next_state

            if state["battery"] < 5 or state["error_rate"] > 0.8:
                break

            time.sleep(sleep_interval)

        rewards.append(episode_reward)

    if return_rewards:
        return rewards

    np.save("rewards_noop_iot.npy", np.array(rewards))
    plt.plot(rewards)
    plt.title("Always No-Op Baseline - IoT")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid()
    plt.savefig("noop_baseline_iot.png")
    plt.show()

if __name__ == "__main__":
    main(num_episodes=100, sleep_interval=0.1)