import numpy as np
from rk4 import rk4

def rka(x, t, tau, err, derivsRK, param):
    """
    Adaptive Runge-Kutta routine.
    :param x: Current value of the dependent variable
    :param t: independent variable (usually time)
    :param tau: step size (usually time step)
    :param err: Desired fractional local truncation error
    :param derivsRK: right hand side of the ODE; derivsRK is the name of the function which returns dx/dt
    Calling format derivsRK (x, t, param).
    :param param: estra parameters passed to derivsRK
    :return:
    xSmall: New value of the dependent variable
    t: New value of the independent variable
    tau: Suggested step size for next call to rka
    """

    # Set initial variables
    tSave, xSave = t, x  # Save initial values for reference
    safe1, safe2 = 0.9, 4.0  # Safety factors for bounds on tau (hard-coded)
    eps = 1.0E-15

    # Loop over maximum number of attempts to satisfy error bound
    xTemp = np.empty(len(x))
    xSmall = np.empty(len(x))
    xBig = np.empty(len(x))