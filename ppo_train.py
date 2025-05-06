from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from linux_kernel_env import KernelTuneEnv

def main():
    # Créer un environnement vectorisé
    print("Initialisation de l'environnement...")
    env = make_vec_env(lambda: KernelTuneEnv(), n_envs=1)

    # Créer le modèle PPO
    print("Création du modèle PPO...")
    model = PPO(
        "MlpPolicy",  # Politique basée sur un réseau de neurones multi-layer perceptron
        env,
        verbose=2,  # Affiche des informations détaillées pendant l'entraînement
        tensorboard_log="./logs/",  # Dossier pour les logs TensorBoard
        n_steps=256,  # Nombre d'étapes avant une mise à jour du modèle
        learning_rate=3e-4,  # Taux d'apprentissage
        gamma=0.99,  # Facteur de discount
        gae_lambda=0.95,  # Facteur pour l'estimation de l'avantage généralisé
        ent_coef=0.02,  # Coefficient pour la régularisation d'entropie
        vf_coef=0.5,  # Coefficient pour la perte de la fonction de valeur
        max_grad_norm=0.5,  # Norme maximale pour le gradient
    )

    # Entraînement
    print("Début de l'entraînement...")
    model.learn(total_timesteps=100000)

    # Sauvegarde du modèle
    print("Sauvegarde du modèle...")
    model.save("ppo_kernel")
    print("Modèle sauvegardé dans ppo_kernel.zip")

if __name__ == "__main__":
    main()