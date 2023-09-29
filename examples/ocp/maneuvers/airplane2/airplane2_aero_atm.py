import numpy as np
import casadi as ca

from giuseppe.utils.examples import Atmosphere1976, create_buffered_conditional_function


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
s_ref = 500.  # ft**2
weight = 34_200.  # lbm
mass = weight / g0  # slug

# CL = CLa alpha
# CD = CD0 + eta CLa alpha**2
lut_data = {
    'M': np.array((0., 0.4, 0.8, 1.2, 1.6, 2., 2.4, 2.8, 3.2)),
    'h': 1e3 * np.array((-2., 0., 5., 15., 25., 35., 45., 55., 65., 75., 85., 95., 105.)),
    'CLa': np.array((2.240, 2.325, 2.350, 2.290, 2.160, 1.950, 1.700, 1.435, 1.250)),
    'CD0': np.array((0.0065, 0.0055, 0.0060, 0.0118, 0.0110, 0.00860, 0.0074, 0.0069, 0.0068)),
    'T': 1e3 * np.array((
        (23.3, 23.3, 20.6, 15.4, 9.9, 5.8, 2.9, 1.3, 0.7, 0.3, 0.1, 0.1, 0.),
        (22.8, 22.8, 19.8, 14.4, 9.9, 6.2, 3.4, 1.7, 1.0, 0.5, 0.3, 0.1, 0.1),
        (24.5, 24.5, 22.0, 16.5, 12.0, 7.9, 4.9, 2.8, 1.6, 0.9, 0.5, 0.3, 0.2),
        (29.4, 29.4, 27.3, 21.0, 15.8, 11.4, 7.2, 3.8, 2.7, 1.6, 0.9, 0.6, 0.4),
        (29.7, 29.7, 29.0, 27.5, 21.8, 15.7, 10.5, 6.5, 3.8, 2.3, 1.4, 0.8, 0.5),
        (29.9, 29.9, 29.4, 28.4, 26.6, 21.2, 14.0, 8.7, 5.1, 3.3, 1.9, 1.0, 0.5),
        (29.9, 29.9, 29.2, 28.4, 27.1, 25.6, 17.2, 10.7, 6.5, 4.1, 2.3, 1.2, 0.5),
        (29.8, 29.8, 29.1, 28.2, 26.8, 25.6, 20.0, 12.2, 7.6, 4.7, 2.8, 1.4, 0.5),
        (29.7, 29.7, 28.9, 27.5, 26.1, 24.9, 20.3, 13.0, 8.0, 4.9, 2.8, 1.4, 0.5),
    ))
}
eta = 1.

# Convert to:
# CD = CD0 + CD2 * CL**2
# NOTE:
# CD2 * CL**2 = (CD2 CLa**2) alpha**2 = eta CLa alpha**2
# So:
# CD2 = eta CLa / CLa**2 = eta / CLa
lut_data['CD2'] = eta / lut_data['CLa']

# Build aero look-up tables
mach_sym = ca.MX.sym('M')
interpolant_CLa = ca.interpolant('CLa', 'bspline', (lut_data['M'],), lut_data['CLa'])
interpolant_CD0 = ca.interpolant('CD0', 'bspline', (lut_data['M'],), lut_data['CD0'])
interpolant_CD2 = ca.interpolant('CD2', 'bspline', (lut_data['M'],), lut_data['CD2'])
interp_fun_CLa = ca.Function('CLa', (mach_sym,), (interpolant_CLa(mach_sym),), ('M',), ('CLa',))
interp_fun_CD0 = ca.Function('CD0', (mach_sym,), (interpolant_CD0(mach_sym),), ('M',), ('CD0',))
interp_fun_CD2 = ca.Function('CD2', (mach_sym,), (interpolant_CD2(mach_sym),), ('M',), ('CD2',))

# Since tables have a discontinuity at the endpoints of Mach number, create buffered conditional function with constant
# extrapolation.
mach_boundary_thickness = 0.05
CLa_expr = create_buffered_conditional_function(
    expr_list=[lut_data['CLa'][0], interp_fun_CLa(mach_sym), lut_data['CLa'][-1]],
    break_points=[-np.inf, lut_data['M'][0] + mach_boundary_thickness, lut_data['M'][-1] - mach_boundary_thickness],
    independent_var=mach_sym,
    boundary_thickness=0.1  # 0.1 Mach boundary thickness
)
CD0_expr = create_buffered_conditional_function(
    expr_list=[lut_data['CD0'][0], interp_fun_CD0(mach_sym), lut_data['CD0'][-1]],
    break_points=[-np.inf, lut_data['M'][0] + mach_boundary_thickness, lut_data['M'][-1] - mach_boundary_thickness],
    independent_var=mach_sym,
    boundary_thickness=0.1  # 0.1 Mach boundary thickness
)
CD2_expr = create_buffered_conditional_function(
    expr_list=[lut_data['CD2'][0], interp_fun_CD2(mach_sym), lut_data['CD2'][-1]],
    break_points=[-np.inf, lut_data['M'][0] + mach_boundary_thickness, lut_data['M'][-1] - mach_boundary_thickness],
    independent_var=mach_sym,
    boundary_thickness=0.1  # 0.1 Mach boundary thickness
)
CLa_fun = ca.Function('CLa', (mach_sym,), (CLa_expr,), ('M',), ('CLa',))
CD0_fun = ca.Function('CD0', (mach_sym,), (CD0_expr,), ('M',), ('CD0',))
CD2_fun = ca.Function('CD2', (mach_sym,), (CD2_expr,), ('M',), ('CD2',))


def max_ld_fun(_CLa, _CD0, _CD2):
    _CD1 = 0.
    _CL0 = 0.
    _CL_max_ld = (_CD0 / _CD2) ** 0.5
    _CD_max_ld = _CD0 + _CD1 * _CL_max_ld + _CD2 * _CL_max_ld ** 2
    _alpha_max_ld = (_CL_max_ld - _CL0) / _CLa
    _dict = {'alpha': _alpha_max_ld, 'CL': _CL_max_ld, 'CD': _CD_max_ld, 'LD': _CL_max_ld / _CD_max_ld}
    return _dict


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

    plt.show()
