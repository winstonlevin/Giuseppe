import os
import pickle

import numpy as np
import matplotlib
from matplotlib import pyplot as plt

import giuseppe

os.chdir(os.path.dirname(__file__))  # Set diectory to current location
matplotlib.use('TkAgg', force=True)

with open('sol_set.data', 'rb') as file:
    sol = pickle.load(file)[-1]
with open('sol_outer.data', 'rb') as file:
    sol_outer = pickle.load(file)


def gen_aux(_sol: giuseppe.Solution):
    _hE = _sol.find('hE')
    _h = _sol.find('h')
    _g = _sol.find('g')
    _W = _sol.find('W')
    _v2 = 2 * _g * (_hE - _h)
    _v = _v2**0.5

    _rho = _sol.find('rho0') * np.exp(-_h / _sol.find('h_ref'))
    _qdyn = 0.5 * _rho * _v2
    _wing_load = _qdyn * _sol.find('Sref')

    _lift = _wing_load * _sol.u[0]
    _cd = _sol.find('CD0') + _sol.find('eta')/_sol.find('CLa') * _sol.u[0]*_sol.u[0]
    _drag = _wing_load * _cd

    return {'D': _drag, 'L': _lift, 'Qdyn': _qdyn, 'rho': _rho, 'g': _g, 'W': _W}


aux = gen_aux(sol)
aux_outer = gen_aux(sol_outer)

labels_x = (r'$h_g$ [kft]', r'$h$ [kft]', r'$\gamma$ [deg]')
plt_gains_x = (1e-3, 1e-3, 180./np.pi)

fig_states, axes_states = plt.subplots(nrows=3, ncols=2)
for idx, axes_x_lam in enumerate(axes_states):
    axes_x_lam[0].grid(zorder=-1)
    axes_x_lam[0].set_xlabel(r'$t$ [s]')
    axes_x_lam[0].set_ylabel(labels_x[idx])
    axes_x_lam[0].plot(sol_outer.t, sol_outer.x[idx, :] * plt_gains_x[idx], '--')
    axes_x_lam[0].plot(sol.t, sol.x[idx, :] * plt_gains_x[idx])

    axes_x_lam[1].grid(zorder=-1)
    axes_x_lam[1].set_xlabel(r'$t$ [s]')
    axes_x_lam[1].set_ylabel(r'$(\lambda)$')
    axes_x_lam[1].plot(sol_outer.t, sol_outer.lam[idx, :], '--')
    axes_x_lam[1].plot(sol.t, sol.lam[idx, :])

fig_states.tight_layout()


def get_u_vals(_sol, _aux):
    return (_sol.u[0], _aux['L'] / _aux['W'], _aux['D'] / _aux['W'])


labels_u = (r'$C_L$ [-]', r'$L$ [g]', r'$D$ [g]')
u_vals = get_u_vals(sol, aux)
u_vals_outer = get_u_vals(sol_outer, aux_outer)

fig_control, axes_control = plt.subplots(nrows=3, ncols=1)
for idx, ax_u in enumerate(axes_control):
    ax_u.grid(zorder=-1)
    ax_u.set_xlabel(r'$t$ [s]')
    ax_u.set_ylabel(labels_u[idx])
    ax_u.plot(sol_outer.t, u_vals_outer[idx], '--')
    ax_u.plot(sol.t, u_vals[idx])

fig_control.tight_layout()

plt.show()
