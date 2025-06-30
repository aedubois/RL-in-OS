import numpy as np

# Chemin de la Q-table
q_table_path = "First Scenario - Desktop/light_first_scenario/q_table.npy"

# Chargement
q_table = np.load(q_table_path)
print(f"Q-table shape: {q_table.shape}")

# Statistiques globales
print(f"Q min: {q_table.min():.3f} | Q max: {q_table.max():.3f} | Q mean: {q_table.mean():.3f} | Q std: {q_table.std():.3f}")

# Nombre d'états et d'actions
n_states = np.prod(q_table.shape[:-1])
n_actions = q_table.shape[-1]
print(f"Nombre d'états: {n_states} | Nombre d'actions: {n_actions}")

# Distribution des valeurs Q
import matplotlib.pyplot as plt
plt.hist(q_table.flatten(), bins=50)
plt.title("Distribution des valeurs Q")
plt.xlabel("Valeur Q")
plt.ylabel("Fréquence")
plt.show()

# États avec la meilleure valeur Q
flat_q = q_table.reshape(-1, n_actions)
best_q = flat_q.max(axis=1)
best_action = flat_q.argmax(axis=1)
top_idx = np.argsort(best_q)[-10:][::-1]
print("\nTop 10 états (indexés) avec meilleure valeur Q et action optimale :")
for idx in top_idx:
    print(f"État {idx} | Q* = {best_q[idx]:.2f} | Action optimale = {best_action[idx]}")

# Optionnel : afficher la meilleure action pour quelques états "basiques"
print("\nAperçu des meilleures actions pour quelques états simples :")
for i in range(5):
    state_idx = [0]* (len(q_table.shape)-2) + [i, 0]  # état avec stress i, autres à 0
    state_idx = tuple(state_idx)
    q_vals = q_table[state_idx]
    print(f"État stress {i} : Q = {q_vals} | Action optimale = {np.argmax(q_vals)}")

# Analyse de la couverture de la Q-table
nonzero_cases = np.count_nonzero(q_table)
total_cases = q_table.size
percent = 100 * nonzero_cases / total_cases
print(f"\nCases découvertes (Q ≠ 0) : {nonzero_cases} / {total_cases} ({percent:.2f}%)")