import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
import matplotlib as mpl
import pickle

with open('sol_set_hl20_max_range.data', 'rb') as f:
    sols = pickle.load(f)
    sol_max_range = sols[-1]

with open('sol_set_hl20_min_time.data', 'rb') as f:
    sols = pickle.load(f)
    sol_min_time = sols[-1]

sol = sol_max_range

# Create Dicts
k_dict = {}
for key, val in zip(sol.annotations.constants, sol.k):
    k_dict[key] = val

CL0 = k_dict['CL0']
CL1 = k_dict['CL1']
CD0 = k_dict['CD0']
CD1 = k_dict['CD1']
CD2 = k_dict['CD2']
s_ref = k_dict['s_ref']
mass = k_dict['mass']
weight0 = k_dict['weight0']

mu = k_dict['mu']
rm = k_dict['rm']
rho0 = k_dict['rho0']
h_ref = k_dict['h_ref']

x_mult = np.array((k_dict['h_scale'], k_dict['theta_scale'], k_dict['v_scale'], k_dict['gam_scale']))
u_mult = k_dict['alpha_scale']
CL_max = 0.5 * CL1

# Minimum time solution must be propagated with cruising angle of attack.
k_h = 1.
h_cruise = k_dict['hf']
idx_h = 0
idx_tha = 1
idx_v = 2
idx_gam = 3


def calc_control(_t, _x):
    # States
    _h = _x[idx_h]
    _tha = _x[idx_tha]
    _v = _x[idx_v]
    _gam = _x[idx_gam]

    _rho = rho0 * np.exp(-_h / h_ref)
    _qdyn_s_ref = 0.5 * _rho * _v**2 * s_ref
    _r = rm + _h
    _g = mu/_r**2

    # Controls [cruise]
    _lift = mass * (_g - _v**2/_r) + k_h * (h_cruise - _h)
    _CL = _lift / _qdyn_s_ref
    _sin = np.minimum(np.maximum(_CL / (0.5 * CL1), -1.), 1.)
    _asin = np.arcsin(_sin)
    _alpha = _asin/2 - CL0/CL1
    return _alpha


def eom(_t, _x, reverse=False):
    # States
    _h = _x[idx_h]
    _tha = _x[idx_tha]
    _v = _x[idx_v]
    _gam = _x[idx_gam]

    _rho = rho0 * np.exp(-_h / h_ref)
    _qdyn_s_ref = 0.5 * _rho * _v**2 * s_ref
    _r = rm + _h
    _g = mu/_r**2

    # Controls [cruise]
    _lift = mass * (_g - _v**2/_r) + k_h * (h_cruise - _h)
    _CL = _lift / _qdyn_s_ref
    _sin = min(max(_CL / (0.5 * CL1), -1.), 1.)
    _asin = np.arcsin(_sin)
    _alpha = _asin/2 - CL0/CL1

    _CL = CL1 * 0.5 * np.sin(2 * (_alpha + CL0/CL1))
    _CD = CD0 - CD1**2/(4*CD2) + CD2 * np.sin(_alpha + CD1/(2*CD2))**2

    _lift = _qdyn_s_ref * _CL
    _drag = _qdyn_s_ref * _CD

    _dh_dt = _v * np.sin(_gam)
    _dtheta_dt = _v/_r * np.cos(_gam)
    _dv_dt = -_drag / mass - _g * np.sin(_gam)
    _dgam_dt = _lift / (mass * _v) + (_v / _r - _g / _v) * np.cos(_gam)
    _dx_dt = np.array((_dh_dt, _dtheta_dt, _dv_dt, _dgam_dt))

    if reverse:
        return -_dx_dt
    else:
        return _dx_dt


def generate_termination_event():
    def termination_event(_t, _x):
        # States
        _h = _x[idx_h]
        _tha = _x[idx_tha]
        _v = _x[idx_v]
        _gam = _x[idx_gam]

        _rho = rho0 * np.exp(-_h / h_ref)
        _qdyn_s_ref = 0.5 * _rho * _v ** 2 * s_ref
        _r = rm + _h
        _g = mu / _r ** 2

        # Controls [cruise]
        _lift = mass * (_g - _v ** 2 / _r) + k_h * (h_cruise - _h)
        _CL = _lift / _qdyn_s_ref

        return _CL - CL_max

    termination_event.terminal = True
    termination_event.direction = 0

    return termination_event


x1 = sol_min_time.x[:, -1] * x_mult
t1 = sol_min_time.t[-1]
ivp_sol = sp.integrate.solve_ivp(
    lambda t, x: eom(t, x, reverse=False), y0=x1, t_span=(t1, np.inf), events=generate_termination_event()
)
alpha_ivp = calc_control(ivp_sol.t, ivp_sol.y)

t_max_range = sol_max_range.t
x_max_range = (sol_max_range.x.T * x_mult).T
alpha_max_range = sol_max_range.u[0, :] * u_mult
t_min_time = np.append(sol_min_time.t, ivp_sol.t)
x_min_time = np.hstack(((sol_min_time.x.T * x_mult).T, ivp_sol.y))
alpha_min_time = np.append(sol_min_time.u[0, :] * u_mult, alpha_ivp)

e_max_range = mu/rm - mu/(rm + x_max_range[idx_h]) + 0.5 * x_max_range[idx_v, :]
qdyn_max_range = 0.5 * rho0 * np.exp(-x_max_range[idx_h]/h_ref) * x_max_range[idx_v, :]**2
CL_max_range = CL1 * 0.5 * np.sin(2 * (alpha_max_range + CL0/CL1))
n_max_range = qdyn_max_range * s_ref * CL_max_range / weight0

e_min_time = mu/rm - mu/(rm + x_min_time[idx_h]) + 0.5 * x_min_time[idx_v, :]
qdyn_min_time = 0.5 * rho0 * np.exp(-x_min_time[idx_h]/h_ref) * x_min_time[idx_v, :]**2
CL_min_time = CL1 * 0.5 * np.sin(2 * (alpha_min_time + CL0/CL1))
n_min_time = qdyn_min_time * s_ref * CL_min_time / weight0

# PLOTTING
s2min = 1. / 60.
r2d = 180. / np.pi
ydata_arr_max_range = np.vstack((x_max_range, t_max_range, n_max_range))
ydata_list_max_range = list(ydata_arr_max_range)
ydata_arr_min_time = np.vstack((x_min_time, t_min_time, n_min_time))
ydata_list_min_time = list(ydata_arr_min_time)

e_label = r'$E$ [m$^2$/s$^2$]'
e_mult = 1.
y_labels = (r'$h$ [m]', r'$\theta$ [deg]', r'$V$ [m/s]', r'$\gamma$ [deg]', r'$t$ [min]', r'$n$ [g]')
y_mult = (1., r2d, 1., r2d, s2min, 1/weight0)

fig = plt.figure()
axes = []

for idx, (y_max_range, y_min_time) in enumerate(zip(ydata_list_max_range, ydata_list_min_time)):
    axes.append(fig.add_subplot(3, 2, idx+1))
    ax = axes[-1]
    ax.grid()
    ax.set_xlabel(e_label)
    ax.set_ylabel(y_labels[idx])

    ax.plot(e_max_range * e_mult, y_max_range * y_mult[idx], label='Max Range')
    ax.plot(e_min_time * e_mult, y_min_time * y_mult[idx], label='Min Time Dive')
