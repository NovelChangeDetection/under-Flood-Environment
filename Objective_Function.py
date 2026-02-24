import numpy as np
from Glob_Vars import Glob_Vars
from Model_SCA_MCA_E_ADDUNet import Model_SCA_MCA_E_ADDUNet


def Objfun_Cls(Soln):
    Images1 = Glob_Vars.Images1
    Images2 = Glob_Vars.Images2
    Label = Glob_Vars.Label
    if Soln.ndim == 2:
        v = Soln.shape[0]
        Fitn = np.zeros((Soln.shape[0], 1))
    else:
        v = 1
        Fitn = np.zeros((1, 1))
    for i in range(v):
        soln = np.array(Soln)

        if soln.ndim == 2:
            sol = Soln[i]
        else:
            sol = Soln
        steps = 200
        Eval = Model_SCA_MCA_E_ADDUNet(Images1, Images2, Label, steps, sol.astype('int'))
        Fitn[i] = 1 / (Eval[20] + Eval[21]) # 1 / (mIoU + SeK)
    return Fitn
