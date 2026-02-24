import time
import numpy as np


def FOA(X, obj_func, lb, ub, max_iter):
    """
    Parameters:
    - obj_func: the objective function to minimize
    - lb: lower bound (can be scalar or array of size dim)
    - ub: upper bound (can be scalar or array of size dim)
    - dim: number of dimensions (features)
    - num_agents: number of fossas
    - max_iter: maximum number of iterations
    """

    # Step 1: Initialize fossa population
    num_agents, dim = X.shape[0], X.shape[1]

    # Evaluate objective function
    F = np.apply_along_axis(obj_func, 1, X)

    # Store the best solution
    best_idx = np.argmin(F)
    best_pos = X[best_idx].copy()
    best_score = F[best_idx]
    conv = np.zeros(max_iter)
    ct  =  time.time()
    # Main loop
    for t in range(max_iter):
        for i in range(num_agents):

            ### -------- Exploration Phase: Attacking the Lemur -------- ###
            # Determine candidate lemurs
            candidate_lemurs_idx = [k for k in range(num_agents) if F[k] < F[i] and k != i]

            if candidate_lemurs_idx:
                # Select one lemur randomly
                selected_lemur_idx = np.random.choice(candidate_lemurs_idx)
                selected_lemur = X[selected_lemur_idx]

                # Eq. (5): Compute new position
                I = np.random.randint(1, 3, dim)  # Either 1 or 2
                r = np.random.rand(dim)
                X_P1 = X[i] + r * (selected_lemur - I * X[i])

                # Clamp to bounds
                X_P1 = np.clip(X_P1, lb, ub)

                # Evaluate
                F_P1 = obj_func(X_P1)

                # Eq. (6): Greedy selection
                if F_P1 <= F[i]:
                    X[i] = X_P1
                    F[i] = F_P1

            ### -------- Exploitation Phase: Chasing the Lemur -------- ###
            r = np.random.rand(dim)
            X_P2 = X[i] + (1 - 2 * r) * ((ub - lb) / t)
            X_P2 = np.clip(X_P2, lb, ub)
            F_P2 = obj_func(X_P2)

            if F_P2 <= F[i]:
                X[i] = X_P2
                F[i] = F_P2

        # Update global best
        min_idx = np.argmin(F)
        if F[min_idx] < best_score:
            best_score = F[min_idx]
            best_pos = X[min_idx].copy()
        conv[t] = best_score

    ct  =  time.time()
    return conv, best_score, best_pos, time
