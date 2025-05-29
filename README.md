# Reinforcement Learning for System Optimization

This repository contains three complete scenarios showcasing how reinforcement learning (Q-learning) can be applied to system-level optimization across different environments:

- **Desktop (interactive usage)**
- **Server (web load + kernel/network tuning)**
- **IoT / Embedded (low-power simulation)**

Each scenario defines its own environment, metrics, reward strategy, and possible actions. All agents use tabular Q-learning and learn to apply the best system parameters depending on observed metrics.

---

## Folder Structure

```
.
├── First Scenario - Desktop/
│   ├── agent.py
│   ├── train_agent.py
│   ├── gui_interface.py
│   ├── monitor_interface.py
│   ├── q_table.npy
│   ├── actions_logs/
│   ├── metrics_logs/
│   ├── plots/
│   └── utils/
│
├── Second Scenario - Server/
│   ├── agent_server.py
│   ├── train_server_agent.py
│   ├── load_generator.py
│   ├── q_table_server.npy│   
│   ├── best_configs.json
│   └── plots/
│
├── Third Scenario - IoT/
│   ├── agent_iot.py
│   ├── train_iot_agent.py
│   ├── q_table_iot.npy
│   └── plots/
│
├── requirements.txt
└── .gitignore
```

---

## Setup

### Python Environment

Install all dependencies using:

```bash
pip install -r requirements.txt
```

### System Requirements

| Scenario   | OS      | Required tools                  | Notes                               |
|------------|---------|----------------------------------|-------------------------------------|
| Desktop    | Linux   | `stress-ng`, `vlc`, `iperf3`, etc. | Root required for full actions     |
| Server     | Linux   | `nginx`, `wrk`                  | Ensure `/server.html` is served    |
| IoT        | Any     | (none, fully simulated)         | Simulates Raspberry Pi environment |

---

## Scenarios Overview

### 1. Desktop (User Workstation)

- Simulates everyday usage: video, browsing, CPU load, etc.
- RL actions: swappiness, CPU governor, process priority, drop caches, etc.
- Goal: keep the system responsive while minimizing CPU and temperature spikes.
- Includes graphical interface to view actions and metrics.

```bash
cd "First Scenario - Desktop"
python3 train_agent.py
```

Or use:

```bash
python3 gui_interface.py
```

---

### 2. Server (Web + Kernel Tuning)

- Synthetic HTTP load using `wrk` against `nginx`.
- RL actions: dirty_ratio, buffer sizes, zswap, process priority.
- Metrics: CPU, IO wait, context switches, network usage, requests/sec.
- Goal: maximize throughput while maintaining low latency and system pressure.

```bash
cd "Second Scenario - Server"
sudo systemctl start nginx
python3 train_server_agent.py
```

---

### 3. IoT / Embedded

- Simulated low-power device performing periodic light tasks.
- RL actions: reduce CPU frequency, reduce disk I/O, dim screen, enable sleep.
- Metrics: temperature, battery level, error rate, disk/network usage.
- Goal: reduce energy use and heat, while maintaining minimal responsiveness.

```bash
cd "Third Scenario - IoT"
python3 train_iot_agent.py
```

---

## Expected Outputs

- **Q-tables**: learned action values for each scenario and metric state
- **Logs**: printed state transitions, actions, rewards
- **Plots**: reward evolution during training (automatically saved in each scenario's `plots/` folder)
- **Best configs** (Server): saved top-performing system configurations

---

## Tips

- Always **run as root** for actions that touch `/sys/` or kernel parameters.
- Adjust **bins, metrics and actions** for each agent if your system differs.
- You can safely **interrupt training**, Q-tables are saved progressively.
- All scenarios can be extended with more actions, continuous state tracking or deep RL variants.

---

## Author

- **[Dubois Alexandre](https://github.com/aedubois)**
