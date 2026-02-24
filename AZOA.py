import time

import numpy as np


def AZOA(X, fitness, lowerbound, upperbound, Max_iterations):
    SearchAgents, dimension = X.shape[0], X.shape[1]
    lowerbound = np.ones(dimension) * lowerbound  # Lower limit for variables
    upperbound = np.ones(dimension) * upperbound  # Upper limit for variables


    fit = np.zeros(SearchAgents)
    for i in range(SearchAgents):
        L = X[i, :]
        fit[i] = fitness(L)

    best_so_far = np.zeros(Max_iterations)
    average = np.zeros(Max_iterations)
    ct = time.time()
    for t in range(Max_iterations):
        # update the global best (fbest)
        best = np.min(fit)
        location = np.argmin(fit)
        if t == 0:
            PZ = X[location, :]  # Optimal location
            fbest = best  # The optimization objective function
        elif best < fbest:
            fbest = best
            PZ = X[location, :]

        # PHASE1: Foraging Behaviour
        for i in range(SearchAgents):
            I = np.round(1 + np.random.rand()).astype(int)
            X_newP1 = X[i, :] + np.random.rand(dimension) * (PZ - I * X[i, :])  # Eq(3)
            X_newP1 = np.maximum(X_newP1, lowerbound)
            X_newP1 = np.minimum(X_newP1, upperbound)

            # Updating X_i using (5)
            f_newP1 = fitness(X_newP1)
            if f_newP1 <= fit[i]:
                X[i, :] = X_newP1
                fit[i] = f_newP1

        # End Phase 1: Foraging Behaviour

        # PHASE2: defense strategies against predators
        Ps = np.random.rand()
        k = np.random.permutation(SearchAgents)[0]
        AZ = X[k, :]  # attacked zebra

        for i in range(SearchAgents):
            if Ps < 0.5:
                # S1: the lion attacks the zebra and thus the zebra chooses an escape strategy
                R = 0.1
                X_newP2 = X[i, :] + R * (2 * np.random.rand(dimension) - 1) * (1 - t / Max_iterations) * X[i,
                                                                                                         :]  # Eq.(5) S1
                X_newP2 = np.maximum(X_newP2, lowerbound)
                X_newP2 = np.minimum(X_newP2, upperbound)
            else:
                # S2: other predators attack the zebra and the zebra will choose the offensive strategy
                I = np.round(1 + np.random.rand()).astype(int)
                X_newP2 = X[i, :] + np.random.rand(dimension) * (AZ - I * X[i, :])  # Eq(5) S2
                X_newP2 = np.maximum(X_newP2, lowerbound)
                X_newP2 = np.minimum(X_newP2, upperbound)

            f_newP2 = fitness(X_newP2)  # Eq (6)
            if f_newP2 <= fit[i]:
                X[i, :] = X_newP2
                fit[i] = f_newP2

        best_so_far[t] = fbest
        average[t] = np.mean(fit)

    Best_score = fbest
    Best_pos = PZ
    ZOA_curve = best_so_far

    ct = time.time() - ct
    return Best_score, ZOA_curve, Best_pos, ct