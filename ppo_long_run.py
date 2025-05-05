import pandas as pd
import matplotlib.pyplot as plt
import time
from stable_baselines3 import PPO
from linux_kernel_env import KernelTuneEnv

def main():
    # Charger le modèle
    print("Chargement du modèle...")
    model = PPO.load("ppo_kernel")

    # Créer l'environnement
    print("Initialisation de l'environnement...")
    env = KernelTuneEnv()

    # Initialisation des logs
    logs = []

    try:
        # Test initial du modèle
        obs, _ = env.reset()
        for _ in range(10):  # Test sur 10 étapes
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                obs, _ = env.reset()

        # Boucle principale
        obs, _ = env.reset()
        step = 0

        while True:
            # Action prise par le modèle
            action, _ = model.predict(obs, deterministic=True)

            # Étape de l'environnement
            obs, reward, terminated, truncated, _ = env.step(action)

            # Sauvegarde des logs
            logs.append({
                "Step": step,
                "Action": int(action),  # Correction : action est un scalaire
                "CPU": float(obs[0]),
                "Swap": float(obs[1]),
                "Load": float(obs[2]),
                "RAM": float(obs[3]),
                "IOwait": float(obs[4]),
                "Latency": float(obs[5]),
                "Reward": float(reward),
            })

            # Réinitialisation si l'environnement est terminé
            if terminated or truncated:
                obs, _ = env.reset()

            step += 1
            time.sleep(1)

    except KeyboardInterrupt:
        # Sauvegarde des résultats
        save_logs(logs)

def save_logs(logs):
    # Sauvegarde des logs dans un fichier CSV
    df = pd.DataFrame(logs)
    df.to_csv("ppo_long_run_log.csv", index=False)
    print("\n Logs sauvegardés dans ppo_long_run_log.csv")

    # Génération d'un graphique
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
    print("Graphique sauvegardé dans ppo_long_run_plot.png")

if __name__ == "__main__":
    main()

