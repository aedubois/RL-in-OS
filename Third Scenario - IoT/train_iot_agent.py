import time
import numpy as np
from agent_iot import IoTAgent
from simulate_iot_load import simulate_periodic_task
import matplotlib.pyplot as plt

def train_iot_agent(num_episodes=50, sleep_interval=5):
    agent = IoTAgent()
    rewards = []
    q_table_path = "Third Scenario - IoT/q_table_iot.npy"

    try:
        for episode in range(num_episodes):
            print(f"\n=== Episode {episode + 1} / {num_episodes} ===")

            # Simulate sensor task (simulate workload)
            simulate_periodic_task()
            time.sleep(1)

            # Observe system state (before action)
            state_before = agent.get_state()
            print(f"[STATE] Before: {state_before}")

            # Select and apply action
            state_tuple = agent.normalize_state(state_before)
            action_idx = agent.select_action(state_tuple)
            agent.apply_action(action_idx)

            # Let the system stabilize
            time.sleep(sleep_interval)

            # Observe new state
            state_after = agent.get_state()
            print(f"[STATE] After: {state_after}")

            # Compute and apply reward
            reward = agent.compute_reward(state_before, state_after)
            agent.learn(state_before, action_idx, reward, state_after)

            # Decay exploration
            agent.exploration_rate = max(0.05, agent.exploration_rate * agent.exploration_decay)
            rewards.append(reward)

        # Save Q-table
        agent.save_q_table(q_table_path)
        print(f"\nQ-table saved to: {q_table_path}")

        # Plot reward evolution
        if len(rewards) >= 2:
            window = min(10, len(rewards))
            moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
            plt.figure(figsize=(10, 5))
            plt.plot(range(window, window + len(moving_avg)), moving_avg, label=f"Moving average (window={window})")
            plt.title("Reward Trend Over Episodes")
            plt.xlabel("Episode")
            plt.ylabel("Reward")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.show()

    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving Q-table...")
        agent.save_q_table(q_table_path)

if __name__ == "__main__":
    train_iot_agent()
