# Third Scenario - IoT

## General Description

This scenario implements a reinforcement learning agent (Q-learning) for managing a simulated IoT-like environment.
The agent operates in a constrained embedded setting, simulating a device similar to a Raspberry Pi, where power efficiency, thermal stability, and minimal I/O activity are prioritized over raw performance.

The environment periodically simulates lightweight tasks (compression, disk writes, sensor readings), and the agent learns to apply system-level energy-saving actions such as CPU governor tuning or brightness reduction.

The environment periodically simulates lightweight tasks (compression, disk writes, sensor readings), and the agent learns to apply system-level energy-saving actions such as CPU governor tuning, sleep mode, or brightness reduction.
Deterministic load spikes are triggered based on specific system states, challenging the agent to anticipate and mitigate their impact.

---

## Folder Structure

Third Scenario - IoT/
│
├── agent_iot.py             # IoT RL agent: state, actions, Q-table, reward, simulation logic, load spikes
├── train_iot_agent.py       # Main RL training loop, learning curve generation
├── random_agent_iot.py      # Random policy baseline
├── heuristic_agent_iot.py   # Heuristic policy baseline
├── noop_policy_iot.py       # No-op (do nothing) baseline
├── compare_strategies_iot.py# Script to compare all strategies and plot results
├── q_table_iot.npy          # (Generated) Q-table save file
├── rewards_random_iot.npy   # (Generated) Rewards for random policy
├── rewards_heuristic_iot.npy# (Generated) Rewards for heuristic policy
├── rewards_noop_iot.npy     # (Generated) Rewards for no-op policy
└── plots/
    └── plot_1.png           # (Generated) Training reward plot(s)

```

```

---

## System Requirements

- **OS**: Linux (Ubuntu or any distro supporting CPU scaling and `/sys/` files)
- **Python**: 3.8+
- No external hardware required (fully simulated)

---

## Python Dependencies

From the root folder:

```bash
pip install -r requirements.txt
```

---

## Usage

### 1. Train the RL agent

```bash
python3 train_iot_agent.py
```

- The agent will simulate an embedded device over multiple episodes.
- It will apply energy-saving actions and attempt to keep the system cool and efficient.
- Rewards are based on both delta (change) and absolute values of temperature, battery, disk IO, etc.
- The Q-table is saved periodically in `q_table_iot.npy`.
- Training performance is plotted and saved in `plots/`.

### 2. Run baselines

- **Random policy**:
  ```bash
  python3 random_agent_iot.py
  ```
- **Heuristic policy**:
  ```bash
  python3 heuristic_agent_iot.py
  ```
- **No-op policy**:
  ```bash
  python3 noop_policy_iot.py
  ```

Each script generates a `.npy` file with episode rewards and a plot.

### 3. Compare strategies

```bash
python3 compare_strategies_iot.py
```

- Runs all four strategies (RL, random, heuristic, no-op) with the same parameters.
- Plots the moving average reward per episode for each policy.
- Prints mean and standard deviation for each strategy.

---

## Main Files

- **agent_iot.py**
  Defines the `IoTAgent` class which simulates the environment, handles state transitions, actions, reward computation, and deterministic load spikes.
- **train_iot_agent.py**
  The main RL training loop. Trains the agent over a number of episodes, applies actions, computes rewards, and plots learning curves.
- **random_agent_iot.py**
  Runs the agent with a random policy for baseline comparison.
- **heuristic_agent_iot.py**
  Runs the agent with a simple rule-based policy for baseline comparison.
- **noop_policy_iot.py**
  Runs the agent with a "do nothing" policy for baseline comparison.
- **compare_strategies_iot.py**
  Runs and compares all strategies, plotting their moving average rewards.
- **q_table_iot.npy**
  Generated automatically. Stores the learned Q-table for later reuse.
- **rewards_*.npy**
  Generated automatically. Stores the episode rewards for each policy.
- **plots/**
  Contains training and comparison reward visualizations.

---

## Key Features

- **Fully simulated**: does not require a real Raspberry Pi or embedded device.
- **Metrics tracked**: CPU frequency, temperature, battery level, disk I/O, error rate, network usage.
- **RL Actions**: powersave CPU, ondemand CPU, reduce writeback interval, enable sleep mode, reduce screen brightness, no-op.
- **Reward function**:

  - Combines both delta (variation) and absolute state penalties/bonuses.
  - Penalizes heat, battery drain, excessive writes, and error spikes.
  - Rewards battery saving, cooling, and stable behavior.
  - Penalizes inappropriate actions (e.g., powersave when not needed).
- **Deterministic load spikes**:

  - Triggered by specific system states (e.g., high temp, high battery, low activity).
  - Only the RL agent can learn to anticipate and mitigate them.
- **Baselines included**: random, heuristic, and no-op for fair comparison.
- **Comparison script**: plots moving averages and prints statistics for all strategies.

---

## Example Output

During training, each episode will show:

- State before/after action
- Executed action
- Reward details
- Early termination on low battery or high error rate

Example terminal logs:

```
[ACTION] Executed: set_cpu_powersave  
[REWARD] ΔTemp=-2.00, ΔBattery=+0.50, ΔDiskIO=-102400, ΔError=+0.01, ΔNet=-15000 | Temp=38.0, Battery=98.5, Error=0.020 → Reward=+1.85
[EVENT] Load spike! (classic)
```

---

## Tips

- **Tune the reward function** in `agent_iot.py` to test different energy strategies.
- **Modify `sleep_interval`** in each script to simulate real time or accelerate learning.
- **Explore action effects** by running fewer steps and observing the simulation behavior closely.
- **Use the saved Q-table** to perform evaluation-only inference.
- **Interrupt safely**: The script saves the Q-table and resets system parameters on keyboard interrupt.
- **Check the comparison plot** to see which strategy performs best in your scenario.
