# Alexander Cox

import gurobipy as gb
from gurobipy import GRB
import numpy as np
import pandas as pd


def process_csv(path):
    data = pd.read_csv(path)

    # remove extra columns

    return np.transpose(data.to_numpy())


if __name__ == "__main__":
    # Create data and set k = # questions, n = # cows, t = # vans
    cows = process_csv("fake.csv")
    k = cows.shape[0]
    n = cows.shape[1]
    t = int(np.ceil(n / 6))

    m = gb.Model('vans')

    # Create matrix of van variables
    vans = []
    for r in range(t):
        van = []
        for c in range(n):
            van.append(m.addVar(vtype=GRB.BINARY, name=f"v_{r},{c}"))
        vans.append(van)

    # Set objective
    # Sum dot product of each van row with each question row
    m.setObjective(np.dot(cows[0], vans[0]) + np.dot(cows[1], vans[0]) + np.dot(cows[2], vans[0])
                   + np.dot(cows[0], vans[1]) + np.dot(cows[1], vans[1]) + np.dot(cows[2], vans[1])
                   # + np.dot(cows[0], vans[2]) + np.dot(cows[1], vans[2]) + np.dot(cows[2], vans[2])
                   # + np.dot(cows[0], vans[3]) + np.dot(cows[1], vans[3]) + np.dot(cows[2], vans[3])
                   , GRB.MAXIMIZE)

    # Set constraints:
    # All rows of vans sum to s, s.t. floor(n / t) <= s <= ceil(n / t)
    for i in range(len(vans)):
        m.addConstr(int(np.floor(n / t)) <= np.sum(vans[i]), name=f"c_1,{i}")
        m.addConstr(int(np.ceil(n / t)) >= np.sum(vans[i]), name=f"c_2,{i}")
    # All columns of vans are identity vectors
    for c in range(len(vans[0])):
        cow = []
        # Construct cow array representing the list of possible van assignments for each cow
        for r in range(len(vans)):
            cow.append(vans[r][c])
        m.addConstr(np.sum(cow) == 1, name=f"c_0,{c}")

    # Optimize model
    m.optimize()

    for v in m.getVars():
        print('%s %g' % (v.VarName, v.X))

    print('Obj: %g' % m.ObjVal)
