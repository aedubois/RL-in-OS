import os
import time
import numpy as np
from agent_server import ServerAgent
from load_generator import run_wrk
from train_server_agent import collect_metrics, reset_sys_params

def heuristic_policy(metrics, agent):
    """Apply a heuristic policy based on system metrics to select an action."""
    if metrics["iowait"] > 1: 
        # If I/O wait is high, increase dirty ratio
        return agent.actions.index("set_dirty_ratio_10")
    if metrics.get("latency", 0) > 1 and metrics["requests_per_sec"] < 100000:
        # If latency is high and requests per second is low, increase memory limits
        return agent.actions.index("set_wmem_max_16M")
    if metrics["cpu_usage"] > 30:
        # If CPU usage is high, reduce dirty ratio
        return agent.actions.index("set_rmem_max_16M")
    if metrics.get("ctx_switches", 0) > 20000:
        # If context switches are high, increase TCP reuse
        return agent.actions.index("set_somaxconn_1024")
    # If none of the conditions are met, do nothing
    return agent.actions.index("no_op")

def main(num_episodes=30, nb_steps_per_episode=10, sleep_interval=1, return_rewards=False):
    """Run a heuristic agent on the server environment for a number of episodes."""
    agent = ServerAgent()
    rewards = []
    for episode in range(num_episodes):
        print(f"\n=== Episode {episode+1} / {num_episodes} ===")
        reset_sys_params()
        previous_actions = []
        requests_per_sec, latency, p99, _ = run_wrk(duration=2)
        state = agent.get_state(collect_metrics(requests_per_sec))
        total_reward = 0
        for step in range(nb_steps_per_episode):
            metrics = collect_metrics(requests_per_sec)
            metrics["latency"] = latency  
            action_idx = heuristic_policy(metrics, agent)
            print("Applying action:", agent.actions[action_idx])
            agent.apply_action(action_idx)
            requests_per_sec, latency, p99, _ = run_wrk(duration=2)
            next_state = agent.get_state(collect_metrics(requests_per_sec))
            metrics = collect_metrics(requests_per_sec)
            reward = agent.compute_reward(metrics, latency=latency, p99=p99)
            if agent.actions[action_idx] != "no_op":
                penalty_factor = agent.penalize_consecutive_actions(action_idx, previous_actions)
                reward *= penalty_factor
            print("reward:", reward)
            previous_actions.append(action_idx)
            total_reward += reward
            state = next_state
            time.sleep(sleep_interval)
        rewards.append(total_reward)
        print(f"Average reward for episode {episode+1}: {total_reward/nb_steps_per_episode}")
    os.makedirs("Second Scenario - Server/rewards", exist_ok=True)
    np.save("Second Scenario - Server/rewards/rewards_heuristic_server.npy", np.array(rewards))
    if return_rewards:
        return rewards

if __name__ == "__main__":
    main()