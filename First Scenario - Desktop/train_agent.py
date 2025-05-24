import time
import random
import subprocess
from agent import EventAgent

NEGATIVE_ACTIONS_INFO = {
    "simulate_cpu_stress":        1,
    "simulate_memory_stress":     2,
    "simulate_disk_fill":         3,
    "simulate_disk_latency":      2,
    "stress_tmpfs":               2,
    "play_streaming_video":       4,
    "simulate_network_stress":    2,
    "simulate_swap_stress":       3,
    "simulate_high_load":         1,
    "simulate_temp_increase":     2
}

NEGATIVE_ACTIONS = list(NEGATIVE_ACTIONS_INFO.keys())

def get_negative_action_delay(action):
    return NEGATIVE_ACTIONS_INFO.get(action, 2)

def apply_negative_action(action):
    """Launches a negative action (system stress) based on the action name."""
    if action == "simulate_cpu_stress":
        print("Simulating CPU stress...")
        return subprocess.Popen("stress-ng --cpu 2 --timeout 8", shell=True)
    elif action == "simulate_memory_stress":
        print("Simulating memory stress...")
        return subprocess.Popen("stress-ng --vm 2 --vm-bytes 1G --timeout 8", shell=True)
    elif action == "simulate_disk_fill":
        print("Simulating disk fill...")
        return subprocess.Popen("dd if=/dev/zero of=/tmp/fillfile bs=1M count=2048", shell=True)
    elif action == "simulate_disk_latency":
        print("Simulating disk latency...")
        return subprocess.Popen("stress-ng --hdd 2 --hdd-bytes 1G --timeout 8", shell=True)
    elif action == "stress_tmpfs":
        print("Simulating tmpfs stress...")
        return subprocess.Popen("dd if=/dev/zero of=/dev/shm/tmpfs_stress bs=1M count=1024", shell=True)
    elif action == "play_streaming_video":
        print("Playing streaming video...")
        return subprocess.Popen(
            "vlc --intf dummy --run-time=8 --play-and-exit https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4 vlc://quit",
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
    elif action == "simulate_network_stress":
        print("Simulating network stress...")
        server_proc = subprocess.Popen("iperf3 -s", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(1)
        client_proc = subprocess.Popen("iperf3 -c 127.0.0.1 -t 8", shell=True)
        return (server_proc, client_proc)
    elif action == "simulate_swap_stress":
        print("Simulating swap stress...")
        return subprocess.Popen("sudo stress-ng --swap 2 --timeout 8", shell=True)
    elif action == "simulate_high_load":
        print("Simulating high load...")
        return subprocess.Popen("timeout 8s yes > /dev/null", shell=True)
    elif action == "simulate_temp_increase":
        print("Simulating temperature increase...")
        return subprocess.Popen("stress-ng --cpu 4 --timeout 8", shell=True)
    else:
        return None

def train_agent(num_episodes=5000, learning_rate=0.1, discount_factor=0.9, exploration_rate=1.0, exploration_decay=0.995):
    """Main training loop for the RL agent."""
    agent = EventAgent()
    agent.learning_rate = learning_rate
    agent.discount_factor = discount_factor
    agent.exploration_rate = exploration_rate

    try:
        for episode in range(num_episodes):
            print(f"Episode {episode+1}/{num_episodes}")
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
            time.sleep(4)

            agent.update_metrics_once()
            new_state = agent.get_normalized_state()

            reward = agent.compute_reward(state, new_state, debug=False)
            agent.learn(state, action_idx, reward, new_state)

            exploration_rate = max(0.05, exploration_rate * exploration_decay)
            agent.exploration_rate = exploration_rate
                
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")

    agent.clean_resources()
    agent.save_q_table("First Scenario - Desktop/q_table.npy")

if __name__ == "__main__":
    train_agent()
