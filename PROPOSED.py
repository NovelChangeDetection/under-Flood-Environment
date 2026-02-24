import time

import numpy as np

def PROPOSED(Gib, obj_func, lb, ub, max_iter=100):
    """
    Wild Gibbon Optimization Algorithm (WGOA) - Modified α implemented.
    :param Gib: Initial population (n_agents, 3, dim)
    :param obj_func: Objective function to minimize
    :param lb: Lower bounds (array or scalar)
    :param ub: Upper bounds (array or scalar)
    :param max_iter: Maximum iterations
    :return: Best solution, fitness, convergence history
    """
    n_agents, dim = Gib.shape[0], Gib.shape[2]

    # Sort memory layers of each family based on fitness
    for i in range(n_agents):
        fit = np.array([obj_func(p) for p in Gib[i]])
        Gib[i] = Gib[i][np.argsort(fit)]

    # Initialize BestGib (global best family)
    best_index = np.argmin([obj_func(Gib[i][0]) for i in range(n_agents)])
    BestGib = Gib[best_index].copy()

    convergence = []
    ct = time.time()
    for t in range(max_iter):
        all_fitness_matrix = np.array([[obj_func(Gib[i][j]) for j in range(3)] for i in range(n_agents)])
        mean_fit = np.mean(all_fitness_matrix)
        worst_fit = np.max(all_fitness_matrix)

        # --- Community Search Strategy ---
        for ii in range(n_agents):
            candidates = []
            for kk in range(3):  # memory layers of family ii
                current = Gib[ii][kk]
                current_fit = obj_func(current)

                for ll in range(3):  # memory layers of BestGib
                    base_alpha = (current + BestGib[ll]) / 2
                    update = mean_fit / (worst_fit + current_fit + 1e-8)
                    alpha = base_alpha + update                            # PROPOSED Updation

                    beta = np.abs(current - BestGib[ll])
                    candidate = np.random.normal(loc=alpha, scale=beta)
                    candidate = np.clip(candidate, lb, ub)
                    candidates.append(candidate)

            # Combine old + new, choose top 3
            combined = np.vstack((Gib[ii], np.array(candidates)))
            fitnesses = np.array([obj_func(p) for p in combined])
            Gib[ii] = combined[np.argsort(fitnesses)[:3]]

        # --- Community Competition Strategy ---
        all_males = np.array([Gib[i][0] for i in range(n_agents)])
        all_fitness = np.array([obj_func(p) for p in all_males])
        best_idx = np.argmin(all_fitness)
        best_candidate = all_males[best_idx]

        if obj_func(best_candidate) < obj_func(BestGib[0]):
            BestGib[0] = best_candidate
            fitnesses = np.array([obj_func(p) for p in BestGib])
            BestGib = BestGib[np.argsort(fitnesses)[:3]]

        convergence.append(obj_func(BestGib[0]))

    ct = time.time() - ct

    return  obj_func(BestGib[0]), convergence, BestGib[0], ct
