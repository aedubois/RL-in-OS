import numpy as np
from linux_kernel_env import KernelTuneEnv

def test_environment():
    """
    Test the KernelTuneEnv environment with random actions.
    """
    # Create an instance of the environment
    env = KernelTuneEnv()

    # Reset the environment
    state, _ = env.reset()
    print(f"Initial state: {state}")

    # Define the maximum number of steps for the test
    max_steps = 10

    for step in range(max_steps):
        # Sample a random action from the action space
        action = env.action_space.sample()
        print(f"\nStep {step + 1}:")
        print(f"Action taken: {action}")

        # Apply the action and retrieve the results
        next_state, reward, terminated, truncated, _ = env.step(action)

        # Display the results
        print(f"Next state: {next_state}")
        print(f"Reward: {reward}")
        print(f"Terminated: {terminated}")
        print(f"Truncated: {truncated}")

        # Check if the episode is terminated or truncated
        if terminated or truncated:
            print("Episode finished!")
            break

    # Clean up resources
    print("Cleaning up resources...")
    env.close()

if __name__ == "__main__":
    test_environment()
