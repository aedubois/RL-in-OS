import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Q-table saved as .npy (adapt the path if needed)
q_table = np.load("Second Scenario - Server/q_table_server.npy")

# If Q-table has more than 2 dimensions, flatten all but the last (actions)
if q_table.ndim > 2:
    heatmap_data = q_table.reshape(-1, q_table.shape[-1])
else:
    heatmap_data = q_table

plt.figure(figsize=(12, 7))
sns.heatmap(heatmap_data, cmap="viridis", cbar=True)
plt.title("Q-table Heatmap (Server Scenario)")
plt.xlabel("Actions")
plt.ylabel("States")
plt.tight_layout()
plt.show()