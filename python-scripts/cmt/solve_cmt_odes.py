import numpy as np
from numpy import conjugate as conj
import scipy as sp
from scipy.integrate import complex_ode
from scipy.constants import c, speed_of_light
from scipy.constants import pi
import matplotlib.pyplot as plt


class myfuncs(object):
    """Talonmies solution to bypass ode_complex bug.
    https://stackoverflow.com/questions/34577870/using-scipy-integrate-complex-ode-instead-of-scipy-integrate-ode?rq=3
    """
    def __init__(self, f, jac=None, fargs=[], jacargs=[]):

        self._f = f
        self._jac = jac
        self.fargs=fargs
        self.jacargs=jacargs

    def f(self, t, y):
        return self._f(t, y, *self.fargs)

    def jac(self, t, y):
        return self._jac(t, y, *self.jacargs)


def coupled_modes(z, y, delta, kappa_a, alpha_a, kappa_b, alpha_b):
    """Equations 4.26 and 4.27 from Okamoto (2006 edition).

    Parameters
    ----------
    A: complex
        Value of field amplitude of mode A at position z.
    B: complex
        Value of field amplitude of mode B at position z.
    delta: complex
        Difference between wavenumbers divided by 2:
        (beta_B - beta_A) / 2
    kappa_a: complex
        Param_kappa_a.
    alpha_a: complex
        Param alpha_a.
    kappa_b: complex
        Param kappa_b.
    alpha_b: complex
        Param alpha_b.

    Returns
    -------
    [complex, complex]
        Array with derivative of A and B wrt z.
    """
    A = y[0]
    B = y[1]
    dA_dz = -1j*kappa_a*B*np.exp(-1j*2*delta*z) + 1j*alpha_a*A
    dB_dz = -1j*kappa_b*A*np.exp(+1j*2*delta*z) + 1j*alpha_b*B
    return [dA_dz, dB_dz]

def calc_params(delta, k12, k21, c12, c21, chi1, chi2):
    denominator = (1 - np.abs(c12)**2) 
    kappa_a = (k12 -c12*chi2) / denominator
    alpha_a = (k21*c12 - chi1) / denominator
    kappa_b = (k12 -conj(c12)*chi1) / denominator
    alpha_b = (k12*conj(c12) -chi2) / denominator
    return (delta, kappa_a, alpha_a, kappa_b, alpha_b)


if __name__ == "__main__":
    # General parameters:
    lambda_0 = 1550e-9 # meters.
    freq = speed_of_light/lambda_0
    beta_0 = 2*pi / lambda_0

    # Coupling parameters: 
    beta_A = 2.5 * beta_0
    beta_B = 2.4 * beta_0
    delta = (beta_B - beta_A) / 2
    k12 = 1.5e5
    k21 = conj(k12)
    c12 = 0.0
    c21 = 0.0
    chi1 = 0.0
    chi2 = 0.0
    fargs = calc_params(delta, k12, k21, c12, c21, chi1, chi2)

    # Spatial coordinates:
    t0 = 0.0 # meters.
    tf = 20e-6 # meters.
    num_points = int(1e4)
    tt = np.linspace(t0, tf, num_points)
    dt = (tf - t0) / num_points

    # Initial values: y0 = [A(0), B(0)]
    y0 = [1+0j, 0j]
    coupled_eqs = myfuncs(f=coupled_modes, fargs=fargs)
    eqs = complex_ode(coupled_eqs.f)
    eqs.set_initial_value(y0, t0)

    AA = []
    BB = []
    while eqs.successful() and eqs.t < tf:
        eqs.integrate(eqs.t+dt)
        # print(eqs.t, eqs.y[0])
        AA.append(eqs.y[0])
        BB.append(eqs.y[1])

    AA = np.array(AA)
    BB = np.array(BB)
    print(eqs.y.shape)

    plt.plot(tt*1e6, np.abs(AA)**2, 'b', label='$P_A$')
    plt.plot(tt*1e6, np.abs(BB)**2, 'g', label='$P_B$')
    plt.legend(loc='best')
    plt.ylabel("P (normalized power)")
    plt.xlabel('$z$' + ' [$\\mu$m]')
    plt.grid()
    plt.show()