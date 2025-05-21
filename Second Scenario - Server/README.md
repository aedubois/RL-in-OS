# Second Scenario - Server

## General Description

This scenario implements a reinforcement learning agent (Q-learning) for dynamic optimisation of system parameters on a Linux server running nginx.
The agent monitors server metrics in real time (CPU, IO, network, RPS, etc.), applies kernel/network tuning actions, and learns to maximise throughput and stability under synthetic HTTP load.

---

## Folder Structure

```
Second Scenario - Server/
│
├── agent_server.py         # RL agent: actions, Q-table, system monitoring (server-specific)
├── train_server_agent.py   # Training script (RL loop, load generation)
├── load_generator.py       # HTTP load generator using wrk, parses metrics
├── q_table_server.npy      # (Generated) Q-table save file
├── best_configs.json       # (Generated) Best configurations found during training
```

---

## Dependencies and Installation

### System Requirements

- Linux (root access required for system tuning)
- nginx (web server)
- wrk (HTTP benchmarking tool)
- Python 3.8+

### Python Dependencies

From the root folder:

```bash
pip install -r requirements.txt
```

### System Tools Installation (example for Ubuntu)

```bash
sudo apt update
sudo apt install nginx wrk
```

---

## Usage

### 1. **Start nginx**

Make sure nginx is running and serving a static file (e.g. `/server.html`):

```bash
sudo systemctl start nginx
```

### 2. **Agent Training (console mode)**

Start RL training with:

```bash
python3 train_server_agent.py
```

- The agent will generate HTTP load, apply system tuning actions, and learn to optimise server performance.
- The Q-table is saved in `q_table_server.npy`.
- The best configurations are saved in `best_configs.json`.

---

## Main Files

- **agent_server.py**:Defines the `ServerAgent` class (monitoring, actions, Q-learning, reward, etc.).
- **train_server_agent.py**:RL training loop, load generation, Q-table and config management.
- **load_generator.py**:Runs wrk, parses HTTP performance metrics (RPS, latency, etc.).
- **q_table_server.npy**:Automatically generated file, contains the saved Q-table.
- **best_configs.json**:
  Automatically generated file, contains the best configurations found.

---

## Expected Results

- **Q-table**: Gradually filled, guides the agent to the best actions for each server state.
- **Best configs**: Top configurations (kernel/network parameters) found during training, exportable as JSON.
- **Plots**: Moving average of reward per episode, visualising learning progress.
- **Console logs**: Actions applied, metrics, and reward details for each episode.

---

## Tips

- **Run scripts as root** to allow all system actions.
- **Adjust bins and episode count** in `agent_server.py` and `train_server_agent.py` according to your server’s power and desired training time.
- **Monitor nginx and wrk** to ensure the server is running and accessible during training.
- **Interrupt safely**: The script saves the Q-table and resets system parameters on keyboard interrupt.
