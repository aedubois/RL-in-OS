import time
import random
from agent import EventAgent, NEGATIVE_ACTIONS, get_negative_action_delay, apply_negative_action

def train_agent(num_episodes=50, nb_steps_per_episode=5, learning_rate=0.1, discount_factor=0.9, exploration_rate=1.0, exploration_decay=0.995):
    """Main training loop for the RL agent."""
    agent = EventAgent()
    agent.learning_rate = learning_rate
    agent.discount_factor = discount_factor
    agent.exploration_rate = exploration_rate

    try:
        for episode in range(num_episodes):
            print(f"\n=== Episode {episode+1}/{num_episodes} ===")
            agent.reset_all_params()
            time.sleep(1)
            total_reward = 0

            for step in range(nb_steps_per_episode):
                negative_action = random.choice(NEGATIVE_ACTIONS)
                proc = apply_negative_action(negative_action)
                delay = get_negative_action_delay(negative_action)
                time.sleep(delay)

                agent.update_metrics_once()
                state = agent.get_normalized_state()

                if negative_action == "simulate_network_stress" and proc is not None:
                    server_proc, client_proc = proc
                    client_proc.wait()
                    server_proc.terminate()
                    server_proc.wait()
                elif proc is not None:
                    proc.wait()

                if random.uniform(0, 1) < exploration_rate:
                    action_idx = random.randint(0, len(agent.actions) - 1)
                else:
                    action_idx = agent.select_action(state)
                agent.apply_action(action_idx)
                time.sleep(2)

                agent.update_metrics_once()
                new_state = agent.get_normalized_state()

                reward = agent.compute_reward(state, new_state, debug=False)
                agent.learn(state, action_idx, reward, new_state)
                total_reward += reward

                state = new_state

            print(f"Total reward for episode {episode+1}: {total_reward:.2f}")

            exploration_rate = max(0.05, exploration_rate * exploration_decay)
            agent.exploration_rate = exploration_rate

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")

    agent.clean_resources()
    agent.save_q_table("First Scenario - Desktop/q_table.npy")

if __name__ == "__main__":
    train_agent()
