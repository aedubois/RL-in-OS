import time
import random
from light_agent import LightEventAgent, NEGATIVE_ACTIONS, get_negative_action_delay, apply_negative_action
import matplotlib.pyplot as plt

def train_agent(num_episodes=200, nb_steps_per_episode=8, learning_rate=0.1, discount_factor=0.9, exploration_rate=1.0, exploration_decay=0.995):
    """Main training loop for the light RL agent in the first scenario."""
    agent = LightEventAgent()
    agent.learning_rate = learning_rate
    agent.discount_factor = discount_factor
    agent.exploration_rate = exploration_rate

    rewards_per_episode = []

    try:
        for episode in range(num_episodes):
            print(f"\n=== Episode {episode+1}/{num_episodes} ===")
            time.sleep(1)
            total_reward = 0

            for step in range(nb_steps_per_episode):
                negative_action = random.choice(NEGATIVE_ACTIONS)
                proc = apply_negative_action(negative_action)
                delay = get_negative_action_delay(negative_action)
                time.sleep(delay)
                agent.last_stress = negative_action

                agent.update_metrics_once()
                state = agent.get_normalized_state()

                if negative_action == "simulate_network_stress" and proc is not None:
                    server_proc, client_proc = proc
                    client_proc.wait()
                    server_proc.terminate()
                    server_proc.wait()
                elif proc is not None:
                    proc.wait()

                if random.uniform(0, 1) < agent.exploration_rate:
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

                print(f"[Step {step+1}] Stress: {negative_action} | Action: {agent.actions[action_idx]} | Reward: {reward:.2f}")

                state = new_state

            print(f"Total reward for episode {episode+1}: {total_reward:.2f}")
            rewards_per_episode.append(total_reward)

            agent.exploration_rate = max(0.05, agent.exploration_rate * exploration_decay)

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")

    agent.clean_resources()
    agent.save_q_table("First Scenario - Desktop/light_first_scenario/q_table.npy")

    plt.plot(rewards_per_episode)
    plt.xlabel("Episode")
    plt.ylabel("Total reward")
    plt.title("Courbe de reward par épisode")
    plt.savefig("First Scenario - Desktop/light_first_scenario/reward_curve.png")
    plt.show()

if __name__ == "__main__":
    train_agent()