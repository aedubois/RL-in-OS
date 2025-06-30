import numpy as np
import matplotlib.pyplot as plt

q_table_path = "First Scenario - Desktop/light_first_scenario/q_table.npy"
q_table = np.load(q_table_path)
print(f"Q-table shape: {q_table.shape}")

print(f"Q min: {q_table.min():.3f} | Q max: {q_table.max():.3f} | Q mean: {q_table.mean():.3f} | Q std: {q_table.std():.3f}")

n_states = np.prod(q_table.shape[:-1])
n_actions = q_table.shape[-1]
print(f"Number of states: {n_states} | Number of actions: {n_actions}")

plt.hist(q_table.flatten(), bins=50)
plt.title("Q-value distribution")
plt.xlabel("Q-value")
plt.ylabel("Frequency")
plt.show()

flat_q = q_table.reshape(-1, n_actions)
best_q = flat_q.max(axis=1)
best_action = flat_q.argmax(axis=1)
top_idx = np.argsort(best_q)[-10:][::-1]
print("\nTop 10 states (indexed) with highest Q-value and optimal action:")
for idx in top_idx:
    print(f"State {idx} | Q* = {best_q[idx]:.2f} | Optimal action = {best_action[idx]}")

print("\nBest actions for a few simple states:")
for i in range(5):
    state_idx = [0] * (len(q_table.shape) - 2) + [i, 0]
    state_idx = tuple(state_idx)
    q_vals = q_table[state_idx]
    print(f"State stress {i}: Q = {q_vals} | Optimal action = {np.argmax(q_vals)}")

nonzero_cases = np.count_nonzero(q_table)
total_cases = q_table.size
percent = 100 * nonzero_cases / total_cases
print(f"\nDiscovered cells (Q ≠ 0): {nonzero_cases} / {total_cases} ({percent:.2f}%)")