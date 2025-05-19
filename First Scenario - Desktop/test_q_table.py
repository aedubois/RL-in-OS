import numpy as np
q = np.load("First Scenario - Desktop/q_table.npy")
print(q.shape)
#print(q)
print("Cases non nulles :", np.count_nonzero(q))
print("Somme Q-table :", np.sum(q))