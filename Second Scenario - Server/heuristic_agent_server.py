import os
import time
import numpy as np
from agent_server import ServerAgent
from load_generator import run_wrk
from train_server_agent import collect_metrics, reset_sys_params

def get_sysctl_value(param):
    """Get the value of a sysctl parameter."""
    try:
        return int(os.popen(f"sysctl {param}").read().split("=")[1].strip())
    except Exception:
        return None

def heuristic_policy(metrics, agent):
    """Apply a heuristic policy based on system metrics to select an action."""

    # dirty_ratio
    dirty_ratio = get_sysctl_value("vm.dirty_ratio")
    if metrics["iowait"] > 1 and dirty_ratio != 10:
        return agent.actions.index("set_dirty_ratio_10")
    if metrics["iowait"] < 0.2 and dirty_ratio != 40:
        return agent.actions.index("set_dirty_ratio_40")

    # wmem_max
    wmem_max = get_sysctl_value("net.core.wmem_max")
    if metrics.get("latency", 0) > 1 and metrics["requests_per_sec"] < 100000 and wmem_max != 16777216:
        return agent.actions.index("set_wmem_max_16M")
    if metrics.get("latency", 0) > 0.5 and wmem_max != 8388608:
        return agent.actions.index("set_wmem_max_8M")

    # rmem_max
    rmem_max = get_sysctl_value("net.core.rmem_max")
    if metrics["cpu_usage"] > 30 and rmem_max != 16777216:
        return agent.actions.index("set_rmem_max_16M")
    if metrics["cpu_usage"] < 10 and rmem_max != 1048576:
        return agent.actions.index("set_rmem_max_1M")

    # ctx_switches / somaxconn
    somaxconn = get_sysctl_value("net.core.somaxconn")
    if metrics.get("ctx_switches", 0) > 20000 and somaxconn != 1024:
        return agent.actions.index("set_somaxconn_1024")
    if metrics.get("ctx_switches", 0) < 5000 and somaxconn != 128:
        return agent.actions.index("set_somaxconn_128")

    # tcp_tw_reuse
    tcp_tw_reuse = get_sysctl_value("net.ipv4.tcp_tw_reuse")
    if metrics.get("latency", 0) > 2 and tcp_tw_reuse != 1:
        return agent.actions.index("set_tcp_tw_reuse_1")
    if metrics.get("latency", 0) < 0.5 and tcp_tw_reuse != 0:
        return agent.actions.index("set_tcp_tw_reuse_0")

    # tcp_fin_timeout
    tcp_fin_timeout = get_sysctl_value("net.ipv4.tcp_fin_timeout")
    if metrics.get("interrupts", 0) > 10000 and tcp_fin_timeout != 10:
        return agent.actions.index("set_tcp_fin_timeout_10")
    if metrics.get("interrupts", 0) < 2000 and tcp_fin_timeout != 60:
        return agent.actions.index("set_tcp_fin_timeout_30")

    # by default, do nothing
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
        last_rps = requests_per_sec
        for step in range(nb_steps_per_episode):
            metrics = collect_metrics(requests_per_sec)
            metrics["latency"] = latency  
            action_idx = heuristic_policy(metrics, agent)
            print("Applying action:", agent.actions[action_idx])
            agent.apply_action(action_idx)
            requests_per_sec, latency, p99, _ = run_wrk(duration=2)
            next_state = agent.get_state(collect_metrics(requests_per_sec))
            metrics = collect_metrics(requests_per_sec)
            reward = agent.compute_reward(metrics, latency=latency, p99=p99, prev_rps=last_rps)
            last_rps = requests_per_sec
            if agent.actions[action_idx] != "no_op":
                penalty_factor = agent.penalize_consecutive_actions(action_idx, previous_actions)
                reward *= penalty_factor
            print("reward:", reward)
            previous_actions.append(action_idx)
            total_reward += reward
            state = next_state
            time.sleep(sleep_interval)
        rewards.append(total_reward/nb_steps_per_episode)
        print(f"Average reward for episode {episode+1}: {total_reward/nb_steps_per_episode}")
    os.makedirs("Second Scenario - Server/rewards", exist_ok=True)
    np.save("Second Scenario - Server/rewards/rewards_heuristic_server.npy", np.array(rewards))
    if return_rewards:
        return rewards

if __name__ == "__main__":
    main()