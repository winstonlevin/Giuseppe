import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
import pickle

DATA = 2

if DATA == 0:
    with open('guess_hl20.data', 'rb') as f:
        sol = pickle.load(f)
        sol.cost = np.nan
        sols = [sol]
elif DATA == 1:
    with open('seed_sol_hl20.data', 'rb') as f:
        sol = pickle.load(f)
        sols = [sol]
elif DATA == 3:
    with open('damned_sols_hl20.data', 'rb') as f:
        sols = pickle.load(f)
        sol = sols[-1]
else:
    with open('sol_set_hl20.data', 'rb') as f:
        sols = pickle.load(f)
        sol = sols[-1]

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

mu = k_dict['mu']
rm = k_dict['rm']
rho0 = k_dict['rho0']
h_ref = k_dict['h_ref']

alpha_max_lift = np.pi/4 - CL0/CL1

idx_h = 0
idx_v = 1
idx_gam = 2


def eom(_t, _x, reverse=False):
    # States
    _h = _x[idx_h]
    _v = _x[idx_v]
    _gam = _x[idx_gam]

    # Controls [apply max lift]
    _alpha = alpha_max_lift

    _rho = rho0 * np.exp(-_h / h_ref)
    _qdyn_s_ref = 0.5 * _rho * _v**2 * s_ref
    _r = rm + _h
    _g = mu/_r**2

    _CL = CL1 * 0.5 * np.sin(2 * (_alpha + CL0/CL1))
    _CD = CD0 - CD1**2/(4*CD2) + CD2 * np.sin(_alpha + CD1/(2*CD2))**2

    _lift = _qdyn_s_ref * _CL
    _drag = _qdyn_s_ref * _CD

    _dh_dt = _v * np.sin(_gam)
    _dv_dt = -_drag / mass - _g * np.sin(_gam)
    _dgam_dt = _lift / (mass * _v) + (_v / _r - _g / _v) * np.cos(_gam)
    _dx_dt = np.array((_dh_dt, _dv_dt, _dgam_dt))

    if reverse:
        return -_dx_dt
    else:
        return _dx_dt


def generate_termination_event(dt_min):
    def termination_event(_t, _x):
        if abs(_t) < dt_min:
            return 1.
        else:
            return _x[idx_gam]

    termination_event.terminal = True
    termination_event.direction = 0

    return termination_event


# Specify Constraints
hf = 10e3
gamf = 0.
gam0 = 0.
pef = mu/rm - mu/(rm + hf)

# Specify range of trial velocities
vf_vals = np.linspace(0.1, 0.5, 10) * 1e3
dt_min = 1e-2

# Propagate solutions
h0_vals = np.empty(vf_vals.shape)
v0_vals = np.empty(vf_vals.shape)
gam0_vals = np.empty(vf_vals.shape)
event_vals = np.empty(vf_vals.shape, dtype=bool)

for idx, vf in enumerate(vf_vals):
    xf = np.array((hf, vf, gamf))
    ef = pef + 0.5 * vf**2
    ivp_sol = sp.integrate.solve_ivp(
        fun=lambda t, x: eom(t, x, reverse=True), t_span=np.array((0., 1e3)), y0=xf,
        events=generate_termination_event(dt_min)
    )

    h0_vals[idx] = ivp_sol.y[idx_h, -1]
    v0_vals[idx] = ivp_sol.y[idx_v, -1]
    gam0_vals[idx] = ivp_sol.y[idx_gam, -1]
    event_vals[idx] = ivp_sol.status == 1


# PLOT
ydata = (h0_vals, v0_vals, gam0_vals)
ylabels = (r'$h_0$', r'$V_0$', r'$\gamma_0$')
xdata = vf_vals
xlabel = (r'$V_f$')
axes = []
ndata = len(ydata)

fig = plt.figure()
for idx, y in enumerate(ydata):
    axes.append(fig.add_subplot(ndata, 1, idx + 1))
    ax = axes[-1]

    ax.plot(xdata, y)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabels[idx])
    ax.grid()

fig.tight_layout()
