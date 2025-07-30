import numpy as np

QTABLE_PATH = "Third Scenario - IoT/q_table_iot.npy"

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
    actions = ["action_1", "action_2", "action_3"]  # Replace with actual action names
    print("\nCorrespondance index → action :")
    for idx, name in enumerate(actions):
        print(f"  {idx}: {name}")
    print("\nAnalyse de la Q-table terminée.")

if __name__ == "__main__":
    main()