import numpy as np
import casadi as ca

from giuseppe.utils.examples import Atmosphere1976, create_buffered_conditional_function


r2d = 180./np.pi
d2r = 1/r2d

# Atmosphere parameters
mu = 0.14076539e17
re = 20_902_900.  # ft
g0 = mu / re ** 2
g = g0
atm = Atmosphere1976(use_metric=False, earth_radius=re, gravity=g0, boundary_thickness=1000.)


h_sym = ca.MX.sym('h')
temp_expr, pres_expr, dens_expr = atm.get_ca_atm_expr(h_sym)
sped_expr = atm.get_ca_speed_of_sound_expr(h_sym)

temp_fun = ca.Function('T', (h_sym,), (temp_expr,), ('h',), ('T',))
pres_fun = ca.Function('P', (h_sym,), (pres_expr,), ('h',), ('P',))
dens_fun = ca.Function('rho', (h_sym,), (dens_expr,), ('h',), ('rho',))
sped_fun = ca.Function('a', (h_sym,), (sped_expr,), ('h',), ('a',))
dens_deriv_fun = ca.Function('drho_dh', (h_sym,), (ca.jacobian(dens_expr, h_sym),), ('h',), ('drho_dh',))

# Aero parameters from Betts' "Practical Methods"
s_ref = 500.  # ft**2
weight0 = 34_200.  # lbm
mass = weight0 / g0  # slug

# CL = CLa alpha
# CD = CD0 + eta CLa alpha**2
lut_data = {
    'M': np.array((0., 0.4, 0.8, 1.2, 1.6, 2., 2.4, 2.8, 3.2)),
    'h': 1e3 * np.array((0., 5., 15., 25., 35., 45., 55., 65., 75., 85., 95., 105.)),
    'CLa': np.array((2.240, 2.325, 2.350, 2.290, 2.160, 1.950, 1.700, 1.435, 1.250)),
    'CD0': np.array((0.0065, 0.0055, 0.0060, 0.0118, 0.0110, 0.00860, 0.0074, 0.0069, 0.0068)),
    'T': 1e3 * np.array((
        (23.3, 20.6, 15.4, 9.9, 5.8, 2.9, 1.3, 0.7, 0.3, 0.1, 0.1, 0.),
        (22.8, 19.8, 14.4, 9.9, 6.2, 3.4, 1.7, 1.0, 0.5, 0.3, 0.1, 0.1),
        (24.5, 22.0, 16.5, 12.0, 7.9, 4.9, 2.8, 1.6, 0.9, 0.5, 0.3, 0.2),
        (29.4, 27.3, 21.0, 15.8, 11.4, 7.2, 3.8, 2.7, 1.6, 0.9, 0.6, 0.4),
        (29.7, 29.0, 27.5, 21.8, 15.7, 10.5, 6.5, 3.8, 2.3, 1.4, 0.8, 0.5),
        (29.9, 29.4, 28.4, 26.6, 21.2, 14.0, 8.7, 5.1, 3.3, 1.9, 1.0, 0.5),
        (29.9, 29.2, 28.4, 27.1, 25.6, 17.2, 10.7, 6.5, 4.1, 2.3, 1.2, 0.5),
        (29.8, 29.1, 28.2, 26.8, 25.6, 20.0, 12.2, 7.6, 4.7, 2.8, 1.4, 0.5),
        (29.7, 28.9, 27.5, 26.1, 24.9, 20.3, 13.0, 8.0, 4.9, 2.8, 1.4, 0.5),
    ))
}
eta = 1.
CD1 = 0.
CL0 = 0.

# Convert aero model to:
# CD = CD0 + CD1 + CD2 * CL**2
# NOTE:
# CD2 * CL**2 = (CD2 CLa**2) alpha**2 = eta CLa alpha**2
# So:
# CD2 = eta CLa / CLa**2 = eta / CLa
lut_data['CD2'] = eta / lut_data['CLa']

# # Convert thrust model to:
# # T = CT * Qdyn * Sref
# h_arr_2d = lut_data['h'].copy().reshape((1, -1))
# lut_data['V'] = lut_data['M'].copy().reshape((-1, 1)) * np.asarray(sped_fun(h_arr_2d))
# lut_data['qdyn'] = 0.5 * np.asarray(dens_fun(h_arr_2d)) * lut_data['V']**2
# lut_data['CT'] = lut_data['T'] / np.maximum(lut_data['qdyn'] * s_ref, 1e-6)

# Saturation functions [since out-of-bounds inputs are ill conditioned for CasADi interpolants]
mach_sym = ca.MX.sym('M')

eps_h = 1e-3
h_min = lut_data['h'][0] + eps_h
h_max = lut_data['h'][-1] - eps_h
h_sat_gain = 100.

h_with_ub = ca.if_else(h_sym < h_max, h_sym, h_max)
h_smooth_with_ub = h_with_ub - ca.log(
            ca.exp(-(h_sym - h_with_ub)*h_sat_gain)
            + np.exp(-(h_max - h_with_ub)*h_sat_gain)
        ) / h_sat_gain
h_with_sub_lb = ca.if_else(h_smooth_with_ub > h_min, h_smooth_with_ub, h_min)
h_sat = h_with_sub_lb + np.log(
            np.exp((h_smooth_with_ub - h_with_sub_lb)*h_sat_gain)
            + np.exp((h_min - h_with_sub_lb)*h_sat_gain)
        ) / h_sat_gain
h_sat_fun = ca.Function('hsat', (h_sym,), (h_sat,), ('h',), ('hsat',))

eps_mach = 1e-3
mach_min = lut_data['M'][0] + eps_mach
mach_max = lut_data['M'][-1] - eps_mach
mach_sat_gain = 100.

mach_with_ub = ca.if_else(mach_sym < mach_max, mach_sym, mach_max)
mach_smooth_with_ub = mach_with_ub - ca.log(
            ca.exp(-(mach_sym - mach_with_ub)*mach_sat_gain)
            + np.exp(-(mach_max - mach_with_ub)*mach_sat_gain)
        ) / mach_sat_gain
mach_with_sub_lb = ca.if_else(mach_smooth_with_ub > mach_min, mach_smooth_with_ub, mach_min)
mach_sat = mach_with_sub_lb + np.log(
            np.exp((mach_smooth_with_ub - mach_with_sub_lb)*mach_sat_gain)
            + np.exp((mach_min - mach_with_sub_lb)*mach_sat_gain)
        ) / mach_sat_gain
mach_sat_fun = ca.Function('Msat', (mach_sym,), (mach_sat,), ('M',), ('Msat',))

# Build aero look-up tables
interpolant_CLa = ca.interpolant('CLa', 'bspline', (lut_data['M'],), lut_data['CLa'])
interpolant_CD0 = ca.interpolant('CD0', 'bspline', (lut_data['M'],), lut_data['CD0'])
interpolant_CD2 = ca.interpolant('CD2', 'bspline', (lut_data['M'],), lut_data['CD2'])
interpolant_thrust = ca.interpolant(
    'thrust_table', 'bspline', (lut_data['M'], lut_data['h']), lut_data['T'].ravel(order='F')
)
CLa_fun = ca.Function('CLa', (mach_sym,), (interpolant_CLa(mach_sat),), ('M',), ('CLa',))
CD0_fun = ca.Function('CD0', (mach_sym,), (interpolant_CD0(mach_sat),), ('M',), ('CD0',))
CD2_fun = ca.Function('CD2', (mach_sym,), (interpolant_CD2(mach_sat),), ('M',), ('CD2',))
thrust_fun = ca.Function('T', (mach_sym, h_sym), (interpolant_thrust(ca.vcat((mach_sat, h_sat))),), ('M', 'h'), ('T',))

# Aerodynamic limits
load_max = 9.
alpha_max = 10. * d2r
qdyn_max = 1e3


def max_ld_fun(_CLa, _CD0, _CD2):
    _CL_max_ld = (_CD0 / _CD2) ** 0.5
    _CD_max_ld = _CD0 + CD1 * _CL_max_ld + _CD2 * _CL_max_ld ** 2
    _alpha_max_ld = (_CL_max_ld - CL0) / _CLa
    _dict = {'alpha': _alpha_max_ld, 'CL': _CL_max_ld, 'CD': _CD_max_ld, 'LD': _CL_max_ld / _CD_max_ld}
    return _dict


def max_ld_fun_mach(_mach):
    if _mach is float:
        _CD0 = float(CD0_fun(_mach))
        _CD2 = float(CD2_fun(_mach))
        _CLa = float(CLa_fun(_mach))
    else:
        _CD0 = np.asarray(CD0_fun(_mach)).flatten()
        _CD2 = np.asarray(CD2_fun(_mach)).flatten()
        _CLa = np.asarray(CLa_fun(_mach)).flatten()

    _CL_max_ld = (_CD0 / _CD2) ** 0.5
    _CD_max_ld = _CD0 + CD1 * _CL_max_ld + _CD2 * _CL_max_ld ** 2
    _alpha_max_ld = (_CL_max_ld - CL0) / _CLa
    _dict = {'alpha': _alpha_max_ld, 'CL': _CL_max_ld, 'CD': _CD_max_ld, 'LD': _CL_max_ld / _CD_max_ld}
    return _dict


def gam_qdyn0(_h, _v):
    _mach = _v / atm.speed_of_sound(_h)
    _ld_max = max_ld_fun_mach(_mach)['LD'][0]
    _beta = - float(dens_deriv_fun(_h)) / atm.density(_h)
    _sin_gam_qdyn0 = -1 / (_ld_max * (1 + _beta * _v**2 / (2*g)))
    return np.arcsin(max(min(_sin_gam_qdyn0, 1.), -1.))


if __name__ == '__main__':
    from matplotlib import pyplot as plt

    mach_vals = np.linspace(-0.25, 3.5, 1000)
    CLa_vals = CLa_fun(mach_vals)
    CD0_vals = CD0_fun(mach_vals)
    CD2_vals = CD2_fun(mach_vals)

    x_data = mach_vals
    x_label = r'$M$'
    y_data = (CLa_vals, CD0_vals, CD2_vals)
    y_labels = (r'$C_{L,a}$', r'$C_{D,0}$', r'$C_{D,2}$')
    n_data = len(y_data)

    fig_coeffs = plt.figure()
    axes = []

    for idx, y in enumerate(y_data):
        axes.append(fig_coeffs.add_subplot(n_data, 1, idx+1))
        ax = axes[-1]
        ax.plot(x_data, y)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_labels[idx])
        ax.grid()

    fig_coeffs.tight_layout()

    mach_vals = np.linspace(mach_min-1., mach_max+1., 1000)
    h_vals = np.linspace(h_min-1e3, h_max+1e3, 1000)
    fig_saturation = plt.figure()

    ax_mach = fig_saturation.add_subplot(121)
    ax_mach.plot(mach_vals, mach_vals, label='Unsaturated')
    ax_mach.plot(mach_vals, np.asarray(mach_sat_fun(mach_vals)).flatten(), label='Saturated')
    ax_mach.plot(mach_vals, 0*mach_vals + mach_min, 'k--', label='Bounds')
    ax_mach.plot(mach_vals, 0*mach_vals + mach_max, 'k--')
    ax_mach.legend()
    ax_mach.set_xlabel('Mach In')
    ax_mach.set_ylabel('Mach Out')
    ax_mach.grid()

    ax_h = fig_saturation.add_subplot(122)
    ax_h.plot(h_vals, h_vals, label='Unsaturated')
    ax_h.plot(h_vals, np.asarray(h_sat_fun(h_vals)).flatten(), label='Saturated')
    ax_h.plot(h_vals, 0*h_vals + h_min, 'k--', label='Bounds')
    ax_h.plot(h_vals, 0*h_vals + h_max, 'k--')
    ax_h.legend()
    ax_h.set_xlabel('h In')
    ax_h.set_ylabel('h Out')
    ax_h.grid()

    fig_saturation.tight_layout()

    plt.show()
