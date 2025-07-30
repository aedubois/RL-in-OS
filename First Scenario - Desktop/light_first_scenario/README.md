# Light First Scenario

This subfolder provides a **lightweight** version of Desktop Scenario.

## Purpose

Train and evaluate a RL agent that can react to various system stresses (CPU, RAM, disk, network, etc.) by applying simple corrective actions, with a smaller codebase and Q-table than the main scenario.

---

## Main files

-**light_agent.py**

  Defines the RL agent, possible stresses (`NEGATIVE_ACTIONS`), system/reaction actions, Q-table management, state discretization, etc.

-**light_train_agent.py**

  Main training loop:

- Generates random stresses
- The agent chooses an action at each step
- Computes reward, updates the Q-table
- Displays and saves the reward curve

-**analyse_q_table.py**

  Script to analyze the generated Q-table:

- Global statistics
- Q-value distribution
- Table coverage (discovered cells)
- Optimal states/actions overview

---

## Key points

-**Simulated stresses:** CPU, memory, disk, disk latency, network, and "no_op" (do nothing).

-**Reaction actions:** system parameter tuning (dirty_ratio, swappiness, read_ahead, zswap...), drop_caches, kill stress processes, "no_op" (do nothing).

-**Q-table:** automatically saved after training, can be analyzed with the provided script.

-**Agent:** discretizes system states, learns via Q-learning.

---

## Quick usage

1.**Train the agent:**

```bash

python3 light_train_agent.py

```

2.**Analyze the Q-table:**

```bash

python3 analyse_q_table.py

```

---

## Who is this for?

- Anyone wanting to quickly test RL strategies on system stresses without the full complexity of the main scenario.
- For exploring, visualizing, and understanding tabular RL agent behavior in a simplified system environment.
