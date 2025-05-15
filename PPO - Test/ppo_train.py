from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from test_ppo.linux_kernel_env import KernelTuneEnv

def main():
    # Create a vectorized environment
    print("Initializing the environment...")
    env = make_vec_env(lambda: KernelTuneEnv(), n_envs=1)

    # Create the PPO model
    print("Creating the PPO model...")
    model = PPO(
        "MlpPolicy",  # Policy based on a multi-layer perceptron neural network
        env,
        verbose=2,  # Display detailed information during training
        tensorboard_log="./logs/",  # Folder for TensorBoard logs
        n_steps=256,  # Number of steps before updating the model
        learning_rate=3e-4,  # Learning rate
        gamma=0.99,  # Discount factor
        gae_lambda=0.95,  # Factor for generalized advantage estimation
        ent_coef=0.02,  # Coefficient for entropy regularization
        vf_coef=0.5,  # Coefficient for value function loss
        max_grad_norm=0.5,  # Maximum gradient norm
    )

    # Training
    print("Starting training...")
    model.learn(total_timesteps=100000)

    # Save the model
    print("Saving the model...")
    model.save("ppo_kernel")
    print("Model saved in ppo_kernel.zip")

if __name__ == "__main__":
    main()