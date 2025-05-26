import time
import numpy as np
from agent_iot import IoTAgent
import matplotlib.pyplot as plt

def train_iot_agent(num_episodes=150, sleep_interval=1):
    agent = IoTAgent()
    rewards = []
    q_table_path = "Third Scenario - IoT/q_table_iot.npy"

    try:
        for episode in range(num_episodes):
            print(f"\n=== Episode {episode + 1} / {num_episodes} ===")
            state = agent.reset()
            episode_reward = 0

            for step in range(100):  # max steps per episode
                print(f"[STATE] {state}")
                state_tuple = agent.normalize_state(state)
                action_idx = agent.select_action(state_tuple)
                agent.apply_action(action_idx)
                next_state = agent.get_state()
                reward = agent.compute_reward(state, next_state)
                agent.learn(state, action_idx, reward, next_state)
                episode_reward += reward
                state = next_state

                # End episode if battery too low or error rate too high
                if state["battery"] < 5 or state["error_rate"] > 0.8:
                    print("[INFO] Battery too low or error rate too high, ending episode.")
                    break

                time.sleep(sleep_interval)

            rewards.append(episode_reward)
            agent.exploration_rate *= agent.exploration_decay

            # Save periodically
            if (episode + 1) % 10 == 0:
                agent.save_q_table(q_table_path)

        # Plot rewards
        plt.plot(rewards)
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title("Learning curve")
        plt.show()

    except KeyboardInterrupt:
        print("Training interrupted.")
        agent.save_q_table(q_table_path)

if __name__ == "__main__":
    train_iot_agent()
