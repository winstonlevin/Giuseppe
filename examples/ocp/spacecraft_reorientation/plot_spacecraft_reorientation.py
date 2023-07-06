import pickle
import matplotlib.pyplot as plt
import numpy as np

DATA = 2

if DATA == 0:
    with open('guess.data', 'rb') as f:
        sol1 = pickle.load(f)
        sol2 = sol1
elif DATA == 1:
    with open('seed_sol.data', 'rb') as f:
        sol1 = pickle.load(f)
        sol2 = sol1
else:
    with open('sol_set_case1.data', 'rb') as f:
        sols = pickle.load(f)
        sol1 = sols[-1]
    with open('sol_set_case2.data', 'rb') as f:
        sols = pickle.load(f)
        sol2 = sols[25]

# Create Dicts
k_dict = {}
x_dict = {}
lam_dict = {}
u_dict = {}

for key, val in zip(sol1.annotations.constants, sol1.k):
    k_dict[key] = val
for key, x_val, lam_val in zip(sol1.annotations.states, list(sol1.x), list(sol1.lam)):
    x_dict[key] = x_val
    lam_dict[key] = lam_val
for key, val in zip(sol1.annotations.controls, list(sol1.u)):
    u_dict[key] = val

eps_u = k_dict['eps_u']
u_min = k_dict['u_min']
u_max = k_dict['u_max']
# u = (u_max - u_min) / np.pi * np.arctan(sol.u / eps_u) + (u_max + u_min)/2
u1 = 0.5 * ((u_max - u_min) * np.sin(sol1.u) + u_max + u_min)
u2 = 0.5 * ((u_max - u_min) * np.sin(sol2.u) + u_max + u_min)

title_str = f'tf(1) = {sol1.t[-1]}, Cost(1) = {sol1.cost}\ntf(2) = {sol2.t[-1]}, Cost(2) = {sol2.cost}'

# PLOT STATES
ylabs = (r'$\omega_1$', r'$\omega_2$', r'$x_1$', r'$x_2$')
fig_states = plt.figure()
axes_states = []

for idx, (state1, state2) in enumerate(zip(list(sol1.x), list(sol2.x))):
    axes_states.append(fig_states.add_subplot(2, 3, idx + 1))
    ax = axes_states[-1]
    ax.grid()
    ax.set_xlabel('Time [s]')
    ax.set_ylabel(ylabs[idx])
    ax.plot(sol1.t, state1)
    ax.plot(sol2.t, state2)

axes_states.append(fig_states.add_subplot(2, 3, 5))
ax = axes_states[-1]
ax.grid()
ax.set_xlabel(ylabs[2])
ax.set_ylabel(ylabs[3])
ax.plot(sol1.x[2, :], sol1.x[3, :])
ax.plot(sol2.x[2, :], sol2.x[3, :])

fig_states.suptitle(title_str)
fig_states.tight_layout()

# PLOT U
ylabs = (r'$u_1$', r'$u_2$')
fig_u = plt.figure()
axes_u = []

for idx, (ctrl1, ctrl2) in enumerate(zip(list(u1), list(u2))):
    axes_u.append(fig_u.add_subplot(2, 2, idx + 1))
    ax = axes_u[-1]
    ax.grid()
    ax.set_xlabel('Time [s]')
    ax.set_ylabel(ylabs[idx])
    ax.plot(sol1.t, ctrl1)
    ax.plot(sol2.t, ctrl2)

fig_u.suptitle(title_str)
fig_u.tight_layout()

plt.show()
