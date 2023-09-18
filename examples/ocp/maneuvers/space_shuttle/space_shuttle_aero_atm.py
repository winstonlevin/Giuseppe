import numpy as np

from giuseppe.utils.examples.atmosphere1976 import Atmosphere1976

r2d = 180./np.pi
d2r = 1/r2d

# Atmosphere parameters
mu = 0.14076539e17
re = 20_902_900.  # ft
g0 = mu / re ** 2
atm = Atmosphere1976(use_metric=False, earth_radius=re, gravity=g0, boundary_thickness=1000.)

# Aero parameters from Betts' "Practical Methods"
s_ref = 2690.  # ft**2
weight = 203_000  # lbm
mass = weight / g0  # slug

# CL = a_0 + a_1 alp_deg
# CD = b_0 + b_1 alp_deg + b_2 alp_deg**2
a_0 = -0.20704
a_1 = 0.029244 * r2d
b_0 = 0.07854
b_1 = -0.61592e-2 * r2d
b_2 = 0.621408e-3 * r2d**2

# Convert to:
# CD = CD0 + CD1 * CL + CD2 * CL**2
CD0 = b_0 - b_1/a_1 * a_0 + b_2 * (a_0/a_1)**2
CD1 = b_1/a_1 - 2 * b_2*a_0/a_1**2
CD2 = b_2/a_1**2





if __name__ == '__main__':
    from matplotlib import pyplot as plt

    alp_vals = np.linspace(-10., 10., 100) * d2r
    CL_vals = a_0 + a_1 * alp_vals
    CD_vals_alp = b_0 + b_1 * alp_vals + b_2 * alp_vals**2
    CD_vals = CD0 + CD1 * CL_vals + CD2 * CL_vals**2

    fig_cd = plt.figure()
    ax_cd = fig_cd.add_subplot(111)
    ax_cd.plot(CL_vals, CD_vals_alp, label='CD(alp)')
    ax_cd.plot(CL_vals, CD_vals, '--', label='CD(CL)')
    ax_cd.set_xlabel('CL')
    ax_cd.set_ylabel('CD')
    ax_cd.legend()

    plt.show()
