import pickle
import matplotlib.pyplot as plt
import numpy as np

DATA = 3

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

# PLOT STATES
ylabs = (r'$\omega_1$', r'$\omega_2$', r'$x_1$', r'$x_2$')
fig_states = plt.figure()
axes_states = []

for idx, state in enumerate(list(sol.x)):
    axes_states.append(fig_states.add_subplot(2, 3, idx + 1))
    ax = axes_states[-1]
    ax.grid()
    ax.set_xlabel('Time [s]')
    ax.set_ylabel(ylabs[idx])
    ax.plot(sol.t, state)

axes_states.append(fig_states.add_subplot(2, 3, 5))
ax = axes_states[-1]
ax.grid()
ax.set_xlabel(ylabs[2])
ax.set_ylabel(ylabs[3])
ax.plot(sol.x[2, :], sol.x[3, :])

fig_states.suptitle(title_str)
fig_states.tight_layout()

# PLOT U
ylabs = (r'$u_1$', r'$u_2$')
fig_u = plt.figure()
axes_u = []

for idx, ctrl in enumerate(list(u)):
    axes_u.append(fig_u.add_subplot(2, 2, idx + 1))
    ax = axes_u[-1]
    ax.grid()
    ax.set_xlabel('Time [s]')
    ax.set_ylabel(ylabs[idx])
    ax.plot(sol.t, ctrl)

fig_u.suptitle(title_str)
fig_u.tight_layout()

plt.show()
