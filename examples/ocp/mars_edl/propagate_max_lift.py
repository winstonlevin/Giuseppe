import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
import matplotlib as mpl
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
gamf = 0.
gam0 = 0.

# Specify range of trial velocities and altitudes
vf_vals = np.linspace(0.5, 0.1, 100) * 1e3
hf_vals = np.linspace(5., 15., 11) * 1e3
dt_min = 1e-2

# Propagate solutions
h0_list = [np.empty(vf_vals.shape)] * hf_vals.shape[0]
v0_list = [np.empty(vf_vals.shape)] * hf_vals.shape[0]
gam0_list = [np.empty(vf_vals.shape)] * hf_vals.shape[0]
event_list = [np.empty(vf_vals.shape)] * hf_vals.shape[0]
ef_list = [np.empty(vf_vals.shape)] * hf_vals.shape[0]
for hf_idx, hf_val in enumerate(hf_vals):
    h0_vals = np.empty(vf_vals.shape)
    v0_vals = np.empty(vf_vals.shape)
    gam0_vals = np.empty(vf_vals.shape)
    ef_vals = np.empty(vf_vals.shape)
    event_vals = np.empty(vf_vals.shape, dtype=bool)

    rf = hf_val + rm
    gf = mu/rf**2

    h0_above_zero = True
    for idx, vf in enumerate(vf_vals):
        qdynf = 0.5 * rho0 * np.exp(-hf_val/h_ref) * vf**2
        lift_max = qdynf * s_ref * CL1 * 0.5
        lift_gam0 = mass * (gf - vf**2/rf)

        if h0_above_zero and lift_max < lift_gam0:
            xf = np.array((hf_val, vf, gamf))
            ef = mu/rm - mu/(rm+hf_val) + 0.5 * vf**2
            ivp_sol = sp.integrate.solve_ivp(
                fun=lambda t, x: eom(t, x, reverse=True), t_span=np.array((0., 1e3)), y0=xf,
                events=generate_termination_event(dt_min)
            )

            h0_vals[idx] = ivp_sol.y[idx_h, -1]
            v0_vals[idx] = ivp_sol.y[idx_v, -1]
            gam0_vals[idx] = ivp_sol.y[idx_gam, -1]
            ef_vals[idx] = ef
            event_vals[idx] = ivp_sol.status == 1

            # Since vf is in descending order, once h0 < 0, we can break.
            h0_above_zero = h0_vals[idx] > 0

            if not event_vals[idx]:
                print('Uh oh!')

        else:
            h0_vals[idx] = np.nan
            v0_vals[idx] = np.nan
            gam0_vals[idx] = np.nan
            ef_vals[idx] = np.nan
            event_vals[idx] = np.nan

    h0_list[hf_idx] = h0_vals
    v0_list[hf_idx] = v0_vals
    gam0_list[hf_idx] = gam0_vals
    ef_list[hf_idx] = ef_vals
    event_list[hf_idx] = event_vals

# Search for approximately optimal solution


# PLOT
gradient = mpl.colormaps['viridis'].colors

# Generate color gradient
if len(h0_list) == 1:
    grad_idcs = np.array((0,), dtype=np.int32)
else:
    grad_idcs = np.int32(np.floor(np.linspace(0, 255, len(h0_list))))


def cols_gradient(n):
    return gradient[grad_idcs[n]]


ylabels = (r'$h_0$', r'$V_0$', r'$\gamma_0$')
ndata = len(ylabels)
xdata = vf_vals
xlabel = (r'$V_f$')
axes = []


fig = plt.figure()
for h_idx, hf_val in enumerate(hf_vals):
    ydata = (h0_list[h_idx], v0_list[h_idx], gam0_list[h_idx])

    for idx, y in enumerate(ydata):
        if h_idx == 0:
            axes.append(fig.add_subplot(ndata, 1, idx + 1))
            ax = axes[idx]
            ax.set_ylabel(ylabels[idx])
            ax.set_xlabel(xlabel)
            ax.grid()

        ax = axes[idx]
        ax.plot(xdata, y)

fig.tight_layout()

fig_paper = plt.figure()
ax_paper = fig_paper.add_subplot(111)
ax_paper.set_ylabel(r'$E_f$ [1,000 m$^2$/s$^2$]')
ax_paper.set_xlabel(r'$h_0$ [km]')
ax_paper.grid()
for h_idx, hf_val in enumerate(hf_vals):
    ax_paper.plot(h0_list[h_idx] / 1e3, ef_list[h_idx] / 1e3, color=cols_gradient(h_idx), label=str(hf_val))

fig_paper.tight_layout()

fig_paper.savefig('hl20_approximate_optimization_sweep.eps', format='eps', bbox_inches='tight')

plt.show()
