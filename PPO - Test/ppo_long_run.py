import pandas as pd
import matplotlib.pyplot as plt
import time
from stable_baselines3 import PPO
from test_ppo.linux_kernel_env import KernelTuneEnv

def main():
    # Load the model
    print("Loading the model...")
    model = PPO.load("ppo_kernel")

    # Create the environment
    print("Initializing the environment...")
    env = KernelTuneEnv()

    # Initialize logs
    logs = []

    try:
        # Initial test of the model
        obs, _ = env.reset()
        for _ in range(10): 
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                obs, _ = env.reset()

        # Main loop
        print("Starting long execution...")
        obs, _ = env.reset()
        step = 0

        while True:
            # Action taken by the model
            action, _ = model.predict(obs, deterministic=True)

            # Step in the environment
            obs, reward, terminated, truncated, _ = env.step(action)

            # Save logs
            logs.append({
                "Step": step,
                "Action": int(action),
                "CPU": float(obs[0]),
                "Swap": float(obs[1]),
                "Load": float(obs[2]),
                "RAM": float(obs[3]),
                "IOwait": float(obs[4]),
                "Latency": float(obs[5]),
                "Reward": float(reward),
            })

            # Reset if the environment is done
            if terminated or truncated:
                obs, _ = env.reset()

            step += 1
            time.sleep(1)

    except KeyboardInterrupt:
        # Save the results
        save_logs(logs)

def save_logs(logs):
    # Save logs to a CSV file
    df = pd.DataFrame(logs)
    df.to_csv("ppo_long_run_log.csv", index=False)
    print("\n Logs saved to ppo_long_run_log.csv")

    # Generate a plot
    plt.figure(figsize=(14, 10))

    metrics = ["CPU", "Swap", "Load", "RAM", "IOwait", "Latency", "Reward"]
    colors = ["blue", "orange", "purple", "red", "green", "brown", "black"]

    for i, metric in enumerate(metrics):
        plt.subplot(len(metrics), 1, i + 1)
        plt.plot(df["Step"], df[metric], color=colors[i])
        plt.ylabel(metric)
        if i == len(metrics) - 1:
            plt.xlabel("Step")

    plt.tight_layout()
    plt.savefig("ppo_long_run_plot.png")
    print("Plot saved to ppo_long_run_plot.png")

if __name__ == "__main__":
    main()
