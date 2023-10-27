import numpy as np
import scipy as sp
from matplotlib import pyplot as plt

r2d = 180. / np.pi

mu = 0.14076539e17
re = 20_902_900.  # ft
g = mu / re ** 2
rho0 = 0.00238
beta = 1/20e3

energy0 = g * 150e3  # ft2/s2
energyf = g * 35e3  # ft2/s2
h_min = 0.05 * energy0 / g
h_max = 0.8 * energy0 / g
h_vals = np.linspace(h_min, h_max, 100)
h_interp_vals = np.linspace(0., energy0/g, 1000)
rho_interp_vals = rho0 * np.exp(-beta * h_interp_vals)

vehicle1_dict = {'W': 30e3, 's_ref': 500, 'CD0': 0.008, 'CD2': 0.5, 'CLa': 0.035 * r2d, 'zeta': 0.1}
vehicle2_dict = {'W': 3e3, 's_ref': 2, 'CD0': 0.220, 'CD2': 0.127, 'CLa': 0.142 * r2d, 'zeta': 0.2}
vehicle1_dict['k_gam_ref'] = 0.1 * vehicle1_dict['CLa']
vehicle2_dict['k_gam_ref'] = 0.2 * vehicle2_dict['CLa']


def find_noc_gains(vehicle_dict):
    CD0 = vehicle_dict['CD0']
    CD2 = vehicle_dict['CD2']
    weight = vehicle_dict['W']
    s_ref = vehicle_dict['s_ref']

    CL_max_ld = (CD0 / CD2)**0.5
    CD_max_ld = CD0 + CD2 * CL_max_ld**2
    k_gam_noc = 2 * (CD_max_ld / (2 * CD2)) ** 0.5

    e_interp_vals = weight / (rho_interp_vals * s_ref * CL_max_ld) + g * h_interp_vals
    h_interp = sp.interpolate.PchipInterpolator(x=e_interp_vals, y=h_interp_vals)

    vehicle_dict['CL_max_ld'] = CL_max_ld
    vehicle_dict['CD_max_ld'] = CD_max_ld
    vehicle_dict['k_gam_noc'] = k_gam_noc
    vehicle_dict['h_interp'] = h_interp


find_noc_gains(vehicle1_dict)
find_noc_gains(vehicle2_dict)


def evaluate_performance(h0, vehicle_dict, k_gam, hit_ground=False, h_ss=False):
    weight = vehicle_dict['W']
    mass = weight / g
    s_ref = vehicle_dict['s_ref']
    CD0 = vehicle_dict['CD0']
    CD2 = vehicle_dict['CD2']
    h_interp = vehicle_dict['h_interp']

    CL_max_ld = (CD0 / CD2)**0.5
    CD_max_ld = CD0 + CD2 * CL_max_ld**2
    E_max_ld = CL_max_ld / CD_max_ld
    qdyn_glide = weight / (s_ref * CL_max_ld)

    if not h_ss:
        def h_fun(_e, _h):
            return _h
    else:
        def h_fun(_e, _h):
            return h_interp(_e)

    # Initial and terminal conditions
    v0 = (2*(energy0 - g*h0)) ** 0.5
    rho_h0 = rho0 * np.exp(-beta * h0)
    sin_gam0 = 1. / (E_max_ld * (1 + beta * qdyn_glide/(rho_h0 * g)))
    gam0 = - np.arcsin(max(min(sin_gam0, 1.), -1.))
    x0 = np.array((h0, 0., v0, gam0))

    if hit_ground:
        def terminal_function(_t, _x):
            _h = _x[0]
            _tha = _x[1]
            _v = _x[2]
            _gam = _x[3]
            return _h
    else:
        def terminal_function(_t, _x):
            _h = _x[0]
            _tha = _x[1]
            _v = _x[2]
            _gam = _x[3]

            _energy = _h * g + 0.5 * _v**2
            return _energy - energyf

    terminal_function.terminal = True
    terminal_function.direction = 0

    def eom(_t, _x):
        _h = _x[0]
        _tha = _x[1]
        _v = _x[2]
        _gam = _x[3]

        _rho = rho0 * np.exp(-beta * _h)
        _qdyn_s_ref = 0.5 * _rho * _v**2 * s_ref

        _e = g * _h + 0.5 * _v**2
        _h_glide = h_fun(_e, _h)
        _rho_inv_glide = np.exp(beta * _h_glide) / rho0

        _sin_gam = 1. / (E_max_ld * (1 + beta * qdyn_glide * _rho_inv_glide/g))
        # _sin_gam = 1. / (E_max_ld * (1 + beta * _v**2 / (2*g)))
        _gam_ss = - np.arcsin(max(min(_sin_gam, 1.), -1.))
        # _gam_ss = 0.
        _CL = CL_max_ld + k_gam * (_gam_ss - _gam)

        _CD = CD0 + CD2 * _CL**2
        _lift = _qdyn_s_ref * _CL
        _drag = _qdyn_s_ref * _CD

        _dh_dt = _v * np.sin(_gam)
        _dtha_dt = _v * np.cos(_gam)
        _dv_dt = - _drag / mass - g * np.sin(_gam)
        _dgam_dt = _lift / (mass * _v) - g/_v * np.cos(_gam)

        return np.array((_dh_dt, _dtha_dt, _dv_dt, _dgam_dt))

    sol_ivp = sp.integrate.solve_ivp(eom, np.array((0., np.Inf)), x0, events=terminal_function)
    t = sol_ivp.t
    x = sol_ivp.y
    tf = t[-1]
    xf = x[:, -1]
    hf = xf[0]
    thaf = xf[1]
    vf = xf[2]
    gamf = xf[3]
    ef = hf*g + 0.5 * vf**2
    dict_f = {'tf': tf, 'Ef': ef, 'hf': hf, 'thaf': thaf, 'vf': vf, 'gamf': gamf, 't': t, 'x': x}
    return dict_f


thaf_1_noc_arr = np.empty(h_vals.shape)
thaf_1_ref_arr = np.empty(h_vals.shape)
thaf_2_noc_arr = np.empty(h_vals.shape)
thaf_2_ref_arr = np.empty(h_vals.shape)

for idx, h_val in enumerate(h_vals):
    thaf_1_noc_arr[idx] = evaluate_performance(h_val, vehicle1_dict, vehicle1_dict['k_gam_noc'])['thaf']
    thaf_1_ref_arr[idx] = evaluate_performance(h_val, vehicle1_dict, vehicle1_dict['k_gam_ref'])['thaf']
    thaf_2_noc_arr[idx] = evaluate_performance(h_val, vehicle2_dict, vehicle2_dict['k_gam_noc'])['thaf']
    thaf_2_ref_arr[idx] = evaluate_performance(h_val, vehicle2_dict, vehicle2_dict['k_gam_ref'])['thaf']

# PLOTTING
cols = plt.rcParams['axes.prop_cycle'].by_key()['color']

fig = plt.figure()
ydata = (thaf_1_noc_arr, thaf_2_noc_arr)
ydata_ref = (thaf_1_ref_arr, thaf_2_ref_arr)
y_mult = 1e-3
x_lab = (r'$h_0$ [1,000 ft]')
y_labs = (r'$x_d$ (Case 1) [1,000 ft]', r'$x_d$ (Case 2) [1,000 ft]')
leg_labs = ('AE/NOC', 'Ref.')

axes = []
for idx, y in enumerate(ydata):
    axes.append(fig.add_subplot(2, 1, idx + 1))
    ax = axes[-1]
    ax.grid()
    ax.plot(h_vals, y * y_mult, color=cols[0], label=y_labs[idx])
    ax.plot(h_vals, 0*y + np.mean(y) * y_mult, '--', color=cols[0])
    ax.plot(h_vals, ydata_ref[idx] * y_mult, color=cols[1], label=y_labs[idx])
    ax.plot(h_vals, 0*ydata_ref[idx] + np.mean(ydata_ref[idx]) * y_mult, '--', color=cols[1])
    ax.set_xlabel(x_lab)
    ax.set_ylabel(y_labs[idx])
axes[0].legend()
fig.tight_layout()

# x_labs = (r'$h$ [kft]', r'$\theta$ [deg]', r'$V$ [ft/s]', r'$\gamma$ [deg]')
# x_mult = (1e-3, r2d, 1., r2d)
# t_lab = r'$t$ [s]'
#
# fig = plt.figure()
# axes = []
# for idx, x in enumerate(dict_11['x']):
#     axes.append(fig.add_subplot(2, 2, idx+1))
#     ax = axes[-1]
#     ax.grid()
#     ax.plot(dict_11['t'], x)
#     ax.set_ylabel(x_labs[idx])
#     ax.set_xlabel(t_lab)
# fig.tight_layout()

plt.show()
