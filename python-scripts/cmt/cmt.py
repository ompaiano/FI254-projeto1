import numpy as np
import scipy as sp



def calc_coupling_length(zz, PA, PB, eta=.5):
    """Calculate the necessary coupling length to transfer eta % of power 
    from mode A to mode B.

    Important: we are assuming the initial conditions
        P_A(z=0) = P_total; 
        P_B(z=0) = 0.

    Parameters
    ----------
    zz: np.array
        Spatial coordinate along propagation direction.
    PA: np.array
        Power in mode A along zz.
    PB: np.array
        Power in mode B along zz.
    eta: float
        Percentage of power transferred from mode A to B.

    Returns
    -------
    Lc: float
        Minimum coupling length required to transfer eta % of power from 
        mode A to mode B.
    """
    PA = PA/PA[0]
    PB = PB/PA[0]
    min_transfer = PB > eta
    idx_Lc = np.argmax(min_transfer)
    return zz[idx_Lc]
