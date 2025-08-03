import time
import random
from light_agent import LightEventAgent, NEGATIVE_ACTIONS, get_negative_action_delay, apply_negative_action

def heuristic_policy(num_episodes=100, nb_steps_per_episode=10, sleep_interval=2):
    """Run a heuristic policy for the LightEventAgent."""
    agent = LightEventAgent()
    rewards_per_episode = []

    try:
        for episode in range(num_episodes):
            print(f"\n=== Heuristic Policy | Episode {episode+1}/{num_episodes} ===")
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

                if negative_action == "simulate_cpu_stress" or negative_action == "simulate_memory_stress":
                    action_name = "drop_caches"
                elif negative_action == "simulate_disk_fill" or negative_action == "simulate_disk_latency":
                    action_name = "set_read_ahead_1024" if "set_read_ahead_1024" in agent.actions else "no_op"
                elif negative_action == "simulate_network_stress":
                    action_name = "enable_zswap" if "enable_zswap" in agent.actions else "no_op"
                else:
                    action_name = "no_op"

                action_idx = agent.actions.index(action_name)
                agent.apply_action(action_idx)
                time.sleep(sleep_interval)

                agent.update_metrics_once()
                new_state = agent.get_normalized_state()

                reward = agent.compute_reward(state, new_state, debug=False)
                total_reward += reward

                print(f"[Step {step+1}] Stress: {negative_action} | Action: {action_name} | Reward: {reward:.2f}")

            print(f"Total reward for episode {episode+1}: {total_reward:.2f}")
            rewards_per_episode.append(total_reward)

    except KeyboardInterrupt:
        print("\nHeuristic policy interrupted by user.")

    agent.clean_resources()

    return rewards_per_episode

if __name__ == "__main__":
    heuristic_policy()