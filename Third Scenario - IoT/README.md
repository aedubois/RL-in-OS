# Third Scenario - IoT

## General Description

This scenario implements a reinforcement learning agent (Q-learning) for managing a simulated IoT-like environment.  
The agent operates in a constrained embedded setting, simulating a device similar to a Raspberry Pi, where power efficiency, thermal stability, and minimal I/O activity are prioritized over raw performance.

The environment periodically simulates lightweight tasks (compression, disk writes, sensor readings), and the agent learns to apply system-level energy-saving actions such as CPU governor tuning or brightness reduction.

---

## Folder Structure

```
Third Scenario - IoT/
│
├── agent_iot.py           # IoT RL agent: state, actions, Q-table, reward, simulation logic
├── train_iot_agent.py     # Main training loop, learning curve generation
├── q_table_iot.npy        # (Generated) Q-table save file
└── plots/
    └── plot_1.png         # (Generated) Training reward plot
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

### 1. Start training

Run the training loop with:

```bash
python3 train_iot_agent.py
```

- The agent will simulate an embedded device over multiple episodes.
- It will apply energy-saving actions and attempt to keep the system cool and efficient.
- Rewards are based on Δ in temperature, battery, disk IO, etc.
- The Q-table is saved periodically in `q_table_iot.npy`.
- Training performance is plotted and saved in `plots/`.

---

## Main Files

- **agent_iot.py**  
  Defines the `IoTAgent` class which simulates the environment, handles state transitions, actions, and reward computation.

- **train_iot_agent.py**  
  The main training loop. Trains the agent over a number of episodes, applies actions, computes rewards, and plots learning curves.

- **q_table_iot.npy**  
  Generated automatically. Stores the learned Q-table for later reuse.

- **plots/**  
  Contains training reward visualizations (one `.png` file per run).

---

## Key Features

- Fully simulated: does not require a real Raspberry Pi or embedded device.
- Metrics tracked: CPU frequency, temperature, battery level, disk I/O, error rate, network usage.
-  RL Actions: powersave CPU, reduce writeback interval, activate sleep mode, reduce brightness, etc.
- Smart reward function:
  - Penalizes heat and excessive writes.
  - Rewards battery saving and stable behavior.
- Handles occasional load spikes and rewards long-term planning.

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
[REWARD] ΔTemp=-2.00, ΔBattery=+0.50, ΔDiskIO=-102400, ΔError=+0.01, ΔNet=-15000 → Reward=+1.85
```

---

## Tips

- You can **tune the reward function** in `agent_iot.py` to test different energy strategies.
- Modify `sleep_interval` in `train_iot_agent.py` to simulate real time or accelerate learning.
- Explore action effects by running fewer steps and observing the simulation behavior closely.
- Use the saved Q-table to perform evaluation-only inference.
- Interrupt safely: The script saves the Q-table and resets system parameters on keyboard interrupt.
