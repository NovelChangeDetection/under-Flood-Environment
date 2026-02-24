import time

import numpy as np


def WGOA(Gib, obj_func, lb, ub, max_iter):
    """
    Wild Gibbon Optimization Algorithm (WGOA) - All formulas implemented.
    :param obj_func: Function to minimize
    :param dim: Number of dimensions
    :param bounds: Tuple (lower_bound, upper_bound)
    :param n_agents: Number of gibbon families (particles)
    :param max_iter: Max number of iterations
    :return: Best solution, fitness, convergence curve
    """
    n_agents, dim = Gib.shape[0], Gib.shape[1]

    # Sort layers (male, female, child) by fitness
    for i in range(n_agents):
        fit = np.array([obj_func(p) for p in Gib[i]])
        sorted_idx = np.argsort(fit)
        Gib[i] = Gib[i][sorted_idx]

    # BestGib initialization (global best)
    best_index = np.argmin([obj_func(Gib[i][0]) for i in range(n_agents)])
    BestGib = Gib[best_index].copy()

    convergence = []
    ct = time.time()
    for t in range(max_iter):
        # COMMUNITY SEARCH STRATEGY
        for ii in range(n_agents):
            candidates = []
            for kk in range(3):
                for ll in range(3):
                    alpha = (Gib[ii][kk] + BestGib[ll]) / 2  # Eq. α
                    beta = np.abs(Gib[ii][kk] - BestGib[ll])  # Eq. β
                    candidate = np.random.normal(loc=alpha, scale=beta)  # GD(α, β)
                    candidate = np.clip(candidate, lb, ub)
                    candidates.append(candidate)
            # Combine old and new positions, pick best 3
            combined = np.vstack((Gib[ii], np.array(candidates)))  # Eq. 2
            fitnesses = np.array([obj_func(p) for p in combined])
            Gib[ii] = combined[np.argsort(fitnesses)[:3]]

        # COMMUNITY COMPETITION STRATEGY
        all_males = np.array([Gib[i][0] for i in range(n_agents)])  # all male gibbons
        all_fitness = np.array([obj_func(p) for p in all_males])
        best_idx = np.argmin(all_fitness)
        best_candidate = all_males[best_idx]

        # Eq. 3: update global best if better
        if obj_func(best_candidate) < obj_func(BestGib[0]):
            BestGib[0] = best_candidate
            fitnesses = np.array([obj_func(p) for p in BestGib])
            BestGib = BestGib[np.argsort(fitnesses)[:3]]

        convergence.append(obj_func(BestGib[0]))


    ct = time.time() - ct
    return obj_func(BestGib[0]), convergence, BestGib[0], ct
