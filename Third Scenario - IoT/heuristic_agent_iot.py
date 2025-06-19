import time
import numpy as np
from agent_iot import IoTAgent
import matplotlib.pyplot as plt
import os

def heuristic_policy(state):
    """A simple heuristic policy for the IoT agent."""
    if state["battery"] < 10:
        return "enable_sleep_mode"
    elif state["temperature"] > 65:
        return "set_cpu_powersave"
    elif state["disk_write_bytes"] > 8e6:
        return "reduce_writeback_interval"
    elif state["error_rate"] > 0.3:
        return "reduce_screen_brightness"
    else:
        return "no_op"

def main(num_episodes=100, sleep_interval=0.1, return_rewards=False):
    """Run a heuristic policy for the IoT agent."""
    agent = IoTAgent()
    rewards = []

    for episode in range(num_episodes):
        print(f"\n=== Episode {episode + 1} / {num_episodes} ===")
        state = agent.reset()
        episode_reward = 0

        for step in range(100):
            action_name = heuristic_policy(state)
            action_idx = agent.actions.index(action_name)
            agent.apply_action(action_idx)
            next_state = agent.get_state()
            reward = agent.compute_reward(state, next_state, action_idx)
            episode_reward += reward
            state = next_state

            if state["battery"] < 5 or state["error_rate"] > 0.8:
                print("[INFO] Battery too low or error rate too high, ending episode.")
                break

            time.sleep(sleep_interval)

        rewards.append(episode_reward)

    if return_rewards:
        return rewards

    np.save("rewards_heuristic_iot.npy", np.array(rewards))
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Heuristic Policy - IoT")
    plt.savefig("heuristic_policy_iot.png")
    plt.show()

if __name__ == "__main__":
    main(num_episodes=100, sleep_interval=0.1)