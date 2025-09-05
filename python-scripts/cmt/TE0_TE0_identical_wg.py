import numpy as np
from numpy import conjugate as conj
import scipy as sp
from scipy.integrate import complex_ode
from scipy.constants import c, speed_of_light
from scipy.constants import pi
import matplotlib.pyplot as plt

# import filetools

import comsol
import solve_cmt_odes
import cmt




directory = ".dados/"
extension = ".csv"
# files = filetools.find.get_files_ending_with(extension, directory)

files = [
    "./dados/coupling_constants_gap_10_1500nm_150pts_sol11.csv",
    "./dados/coupling_constants_gap_10_1500nm_150pts_sol22.csv",
    "./dados/coupling_constants_gap_10_1500nm_150pts_sol12.csv",
    "./dados/coupling_constants_gap_10_1500nm_150pts_sol21.csv"
]

files = ["../../comsol/cmt/cmt_TE0_TE0_guias_identicos_sol22_mais_gaps.csv"]
files = ["../../comsol/cmt/cmt_TM0-TM0_guias_identicos_sol11.csv"]

dfs = []
for f in files:
    dfs.append(comsol.read_csv_from_comsol(f))

# print(df11.columns)

plot = False


for df in dfs:
    neff = 1.4988027009900051 # 
    neff_tol = 1e-4
    df = df[(neff - neff_tol < df.neff.values.real) & (df.neff.values.real < neff + neff_tol)]
    gaps = df.gap.values
    kappa_12 = df.K12.values
    chi11 = df.X11.values
    chi22 = df.X22.values
    c_butt_12 = df.C12.values

    Lc = []

    gap_tol = 1e-9
    for gap in gaps:
        # General parameters:
        lambda_0 = 1550e-9 # meters.
        freq = speed_of_light/lambda_0
        beta_0 = 2*pi / lambda_0

        # Coupling parameters: 
        beta_A = neff * beta_0
        beta_B = neff * beta_0
        delta = (beta_B - beta_A) / 2
        k12 = df.K12[(gap - gap_tol < df.gap) & (df.gap < gap + gap_tol) & (neff - neff_tol < df.neff) & (df.neff < neff + neff_tol)].values[0]
        k21 = conj(k12)
        c12 = df.C12[(gap - gap_tol < df.gap) & (df.gap < gap + gap_tol) & (neff - neff_tol < df.neff) & (df.neff < neff + neff_tol)].values[0]
        c21 = conj(c12)
        chi1 = df.X11[(gap - gap_tol < df.gap) & (df.gap < gap + gap_tol) & (neff - neff_tol < df.neff) & (df.neff < neff + neff_tol)].values[0]
        chi2 = df.X22[(gap - gap_tol < df.gap) & (df.gap < gap + gap_tol) & (neff - neff_tol < df.neff) & (df.neff < neff + neff_tol)].values[0]
        args = [delta, k12, k21, c12, c21, chi1, chi2]
        print(args)
        fargs = solve_cmt_odes.calc_params(delta, k12, k21, c12, c21, chi1, chi2)

        # Spatial coordinates:
        t0 = 0.0 # meters.
        tf = 50e-6 * gap/20e-9 # USE THIS FOR THE TE0 mode: np.exp(gap/100e-9) # meters.
        num_points = int(1e5)
        tt = np.linspace(t0, tf, num_points)
        dt = (tf - t0) / num_points

        # Initial values: y0 = [A(0), B(0)]
        y0 = [1+0j, 0j]
        coupled_eqs = solve_cmt_odes.myfuncs(f=solve_cmt_odes.coupled_modes, fargs=fargs)
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

        Lc.append(cmt.calc_coupling_length(tt, np.abs(AA)**2, np.abs(BB)**2, eta=.5))

        print("Coupling length = ", Lc[-1]*1e6, " micrometers.")

        if plot:
            plt.plot(tt*1e6, np.abs(AA[:len(tt)])**2, 'b', label='$P_A$')
            plt.plot(tt*1e6, np.abs(BB[:len(tt)])**2, 'g', label='$P_B$')
            plt.legend(loc='best')
            plt.ylabel("P (normalized power)")
            plt.xlabel('$z$' + ' [$\\mu$m]')
            plt.grid()
            plt.show()

    Lc = np.array(Lc)

plt.loglog(gaps*1e9, kappa_12.real, label="real[$\\kappa_{12}$]")
plt.loglog(gaps*1e9, c_butt_12.real, label="real[$c_{12}$]")
plt.loglog(gaps*1e9, chi11, label="real[$\\chi_{11}$]")
plt.xlim(10, 1e3)
plt.xlabel("gap [nm]")
plt.ylabel("Coupling constants")
plt.minorticks_on()
plt.grid(which="both")
plt.legend()
plt.savefig("TM_TM_identical_wg_coupling_constants.pdf")
plt.show()

plt.loglog(gaps*1e9, Lc*1e6)
plt.xlim(10, 1e3)
plt.ylim(1, 1e4)
plt.xlabel("gap [nm]")
plt.ylabel("Coupling length [$\\mu$m]")
plt.minorticks_on()
plt.grid(which="both")
plt.savefig("TM_TM_identical_wg_Lc_vs_gap.pdf")
plt.show()