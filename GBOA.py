import time
import numpy as np

def GBOA(P, fobj, lb, ub,  itermax):
    NP, dim = P.shape[0], P.shape[1]
    AllPositionsBest = np.zeros((1, dim, itermax))
    AllPositions = np.zeros((NP, dim, itermax))
    FT = np.zeros(NP)

    for i in range(NP):
        P[i, :] = lb + np.random.rand(1, dim) * (ub - lb)  # Random initial position
        FT[i] = fobj(P[i, :])  # Fitness

    V = np.sort(FT)  # Sorting the fitness value
    Ix = np.argsort(FT)  # Indices of sorted fitness
    PS = P[Ix, :]  # Sorting the position
    FT1 = V  # Sorted fitness value
    pbest = PS[0, :]  # Best position
    G = V[0]  # Best fitness
    pl = 7  # Penis length

    AllPositions[:, :, 0] = PS
    AllPositionsBest[:, :, 0] = pbest
    ct = time.time()
    for itr in range(1, itermax):
        d = np.random.permutation(NP)
        m = np.random.permutation(NP)
        Q = np.zeros((NP, dim))

        for i in range(NP):
            if abs(d[i] - m[i]) <= pl:
                prd = np.random.randn(1, dim)
                qrd = 1 - prd
                Q[i, :] = prd * PS[d[i], :] + qrd * PS[m[i], :]
            else:
                Q[i, :] = np.random.rand() * PS[m[i], :]
            Q[i, :] = np.minimum(Q[i, :], ub)
            Q[i, :] = np.maximum(Q[i, :], lb)
            T = Q[i, :]
            FT2[i] = fobj(T)

        V, ix = np.sort(FT2), np.argsort(FT2)
        X1 = Q[ix, :]

        S1 = X1[:NP // 3, :]
        S2 = np.zeros_like(S1)
        S3 = np.zeros_like(S1)
        lb1 = np.ones((1, dim)) * lb  # lower boundary
        ub1 = np.ones((1, dim)) * ub  # upper boundary

        for i in range(NP // 3):
            S2[i, :] = S1[i, :] * (1 + np.random.randn(1, 1))
            S3[i, :] = pbest - np.mean(S1, axis=0) + np.random.rand() * (lb1 + np.random.rand() * (ub1 - lb1))

        X1[NP // 3:] = np.vstack((S2, S3))
        Q = X1

        for i in range(NP):
            FT2[i] = fobj(Q[i, :])

        FT = np.concatenate((FT1, FT2))
        PQ = np.vstack((PS, Q))
        V, Ix = np.sort(FT), np.argsort(FT)
        PS = PQ[Ix[:NP], :]
        FT1 = V[:NP]
        FT2 = np.zeros(NP)
        Q = np.zeros_like(Q)

        if V[0] < G:
            G = V[0]
            pbest = PS[0, :]
        else:
            G = G

        AllPositions[:, :, itr] = PS
        AllPositionsBest[:, :, itr] = pbest

    Convergence_curve = G
    Destination_fitness = G
    bestPositions = pbest
    ct = time.time() - ct
    return Destination_fitness, Convergence_curve,  bestPositions, ct


