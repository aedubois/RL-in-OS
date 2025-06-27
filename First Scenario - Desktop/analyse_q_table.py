import numpy as np
from agent import get_param_actions, get_reaction_actions

QTABLE_PATH = "First Scenario - Desktop/q_table.npy"

def main():
    q_table = np.load(QTABLE_PATH)
    print(f"Forme de la Q-table : {q_table.shape}")
    print(f"Valeur min : {q_table.min():.4f}")
    print(f"Valeur max : {q_table.max():.4f}")
    print(f"Valeur moyenne : {q_table.mean():.4f}")
    print(f"Nombre de cases non nulles : {(q_table != 0).sum()} / {q_table.size}")

    best_actions = np.argmax(q_table, axis=-1)
    unique, counts = np.unique(best_actions, return_counts=True)
    print("\nActions les plus souvent choisies comme optimales :")
    for action, count in zip(unique, counts):
        print(f"  Action {action} : {count} états")

    actions = [a[0] for a in get_param_actions()] + get_reaction_actions()
    print("\nCorrespondance index → action :")
    for idx, name in enumerate(actions):
        print(f"  {idx}: {name}")

if __name__ == "__main__":
    main()