import pickle
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

mpl.rcParams['axes.formatter.useoffset'] = False

PLOTAUX = True
DATA = 1

if DATA == 0:
    with open('guess.data', 'rb') as f:
        sol = pickle.load(f)
elif DATA == 1:
    with open('seed_sol.data', 'rb') as f:
        sol = pickle.load(f)
elif DATA == 2:
    with open('sol_set_case1.data', 'rb') as f:
        sols = pickle.load(f)
        sol = sols[-1]
elif DATA == 3:
    with open('sol_set_case2.data', 'rb') as f:
        sols = pickle.load(f)
        sol = sols[-1]

# Create Dicts
k_dict = {}
x_dict = {}
lam_dict = {}
u_dict = {}

for key, val in zip(sol.annotations.constants, sol.k):
    k_dict[key] = val
for key, x_val, lam_val in zip(sol.annotations.states, list(sol.x), list(sol.lam)):
    x_dict[key] = x_val
    lam_dict[key] = lam_val
for key, val in zip(sol.annotations.controls, list(sol.u)):
    u_dict[key] = val

eps_u = k_dict['eps_u']
u_min = k_dict['u_min']
u_max = k_dict['u_max']
# u = (u_max - u_min) / np.pi * np.arctan(sol.u / eps_u) + (u_max + u_min)/2
u = 0.5 * ((u_max - u_min) * np.sin(sol.u) + u_max + u_min)

title_str = f'tf = {sol.t[-1]}, Cost = {sol.cost}'


if PLOTAUX:
    q = sol.x[0:4, :]
    qnorm = np.linalg.norm(q, axis=0)
    for (_q, _qnorm) in zip(list(q.T), qnorm):
        _s = 1/_qnorm**2
        _qr = _q[3]
        _qi = _q[0]
        _qj = _q[1]
        _qk = _q[2]
        _R = np.vstack((
            (1 - 2*_s*(_qj**2 + _qk**2), 2 * _s * (_qi*_qj - _qk*_qr), 2*_s*(_qi*_qk+_qj*_qr)),
            (2*_s*(_qi*_qj + _qk*_qr), 1 - 2*_s * (_qi**2 + _qk**2), 2 * _s * (_qj*_qk - _qi*_qr)),
            (2*_s*(_qi*_qk - _qj*_qr), 2*_s*(_qi*_qk + _qi*_qr), 1 - 2*_s * (_qi**2 + _qj**2))
        ))
        _x = _R @ np.vstack((1., 0., 0.))


# PLOT STATES
ylabs = (r'$q_1$', r'$q_2$', r'$q_3$', r'$q_4$', r'$\omega_1$', r'$\omega_2$', r'$\omega_3$')
fig_states = plt.figure()
axes_states = []

for idx, state in enumerate(list(sol.x)):
    axes_states.append(fig_states.add_subplot(4, 2, idx + 1))
    ax = axes_states[-1]
    ax.grid()
    ax.set_xlabel('Time [s]')
    ax.set_ylabel(ylabs[idx])
    ax.plot(sol.t, state)

fig_states.tight_layout()

# PLOT CONTROLS
ylabs = (r'$u_1$', r'$u_2$', r'$u_3$')
fig_control = plt.figure()
axes_control = []

for idx, control in enumerate(list(u)):
    axes_control.append(fig_control.add_subplot(3, 1, idx + 1))
    ax = axes_control[-1]
    ax.grid()
    ax.set_xlabel('Time [s]')
    ax.set_ylabel(ylabs[idx])
    ax.plot(sol.t, control)

fig_control.tight_layout()

# PLOT COSTATES
ylabs = (r'$\lambda_{q_1}$', r'$\lambda_{q_2}$', r'$\lambda_{q_3}$', r'$\lambda_{q_4}$',
         r'$\lambda_{\omega_1}$', r'$\lambda_{\omega_2}$', r'$\lambda_{\omega_3}$')
fig_costates = plt.figure()
axes_costates = []

for idx, costate in enumerate(list(sol.lam)):
    axes_costates.append(fig_costates.add_subplot(4, 2, idx + 1))
    ax = axes_costates[-1]
    ax.grid()
    ax.set_xlabel('Time [s]')
    ax.set_ylabel(ylabs[idx])
    ax.plot(sol.t, costate)

fig_costates.tight_layout()

plt.show()
