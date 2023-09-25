import numpy as np
import casadi as ca

from giuseppe.utils.examples.atmosphere1976 import Atmosphere1976

r2d = 180./np.pi
d2r = 1/r2d

# Atmosphere parameters
mu = 0.14076539e17
re = 20_902_900.  # ft
g0 = mu / re ** 2
atm = Atmosphere1976(use_metric=False, earth_radius=re, gravity=g0, boundary_thickness=1000.)


h_sym = ca.SX.sym('h')
temp_expr, pres_expr, dens_expr = atm.get_ca_atm_expr(h_sym)
sped_expr = atm.get_ca_speed_of_sound_expr(h_sym)

temp_fun = ca.Function('T', (h_sym,), (temp_expr,), ('h',), ('T',))
pres_fun = ca.Function('P', (h_sym,), (pres_expr,), ('h',), ('P',))
dens_fun = ca.Function('rho', (h_sym,), (dens_expr,), ('h',), ('rho',))
sped_fun = ca.Function('a', (h_sym,), (sped_expr,), ('h',), ('a',))

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

CL0 = a_0
CLa = a_1

CL_max_ld = (CD0 / CD2) ** 0.5
CD_max_ld = CD0 + CD1 * CL_max_ld + CD2 * CL_max_ld ** 2
alpha_max_ld = (CL_max_ld - CL0) / CLa


if __name__ == '__main__':
    from matplotlib import pyplot as plt

    alp_vals = np.linspace(-10., 20., 100) * d2r
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

    fig_ld = plt.figure()
    ax_ld = fig_ld.add_subplot(111)
    ax_ld.plot(alp_vals * r2d, CL_vals / CD_vals)
    ax_ld.plot(alpha_max_ld * r2d, CL_max_ld / CD_max_ld, '*', label='Max L/D')
    ax_ld.set_xlabel(r'$\alpha$ [deg]')
    ax_ld.set_ylabel(r'$C_L / C_D$')
    ax_ld.grid()
    ax_ld.legend()

    plt.show()
