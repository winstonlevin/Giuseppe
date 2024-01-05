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
y_dict = {}

for key, val in zip(sol.annotations.constants, sol.k):
    k_dict[key] = val
for key, y_val in zip(sol.annotations.states, list(sol.x)):
    y_dict[key] = y_val

eps_u = k_dict['eps_u']
I = np.array((k_dict['I1'], k_dict['I2'], k_dict['I3'])).reshape((3,1))
q = sol.x[:4, :]
w = sol.x[4:7, :]
lam_q = sol.x[7:11, :]
lam_w = sol.x[11:, :]

# Optimal Control Law
u = (-lam_w / I / (eps_u ** 2 + (lam_w / I) ** 2) ** 0.5)

title_str = f'tf = {sol.t[-1]}, Cost = {sol.cost}'

# PLOT ORIENTATIONS
ylabs = (r'$q_1$', r'$q_2$', r'$q_3$', r'$q_4$')
fig_states = plt.figure()
axes_states = []

for idx, state in enumerate(list(q)):
    axes_states.append(fig_states.add_subplot(2, 2, idx + 1))
    ax = axes_states[-1]
    ax.grid()
    ax.set_xlabel('Time [s]')
    ax.set_ylabel(ylabs[idx])
    ax.plot(sol.t, state)

fig_states.tight_layout()

# PLOT VELOCITIES AND CONTROLS
ylabs = (r'$\omega_1$', r'$\omega_2$', r'$\omega_3$', r'$u_1$', r'$u_2$', r'$u_3$')
ydata = np.vstack((w, u))
plt_idx = (1, 3, 5, 2, 4, 6)
fig_control = plt.figure()
axes_control = []

for idx, y in enumerate(ydata):
    axes_control.append(fig_control.add_subplot(3, 2, plt_idx[idx]))
    ax = axes_control[-1]
    ax.grid()
    ax.set_xlabel('Time [s]')
    ax.set_ylabel(ylabs[idx])
    ax.plot(sol.t, y)

fig_control.tight_layout()

# PLOT COSTATES
ylabs = (r'$\lambda_{q_1}$', r'$\lambda_{q_2}$', r'$\lambda_{q_3}$', r'$\lambda_{q_4}$',
         r'$\lambda_{\omega_1}$', r'$\lambda_{\omega_2}$', r'$\lambda_{\omega_3}$')
fig_costates = plt.figure()
axes_costates = []

for idx, costate in enumerate(list(np.vstack((lam_q, lam_w)))):
    axes_costates.append(fig_costates.add_subplot(4, 2, idx + 1))
    ax = axes_costates[-1]
    ax.grid()
    ax.set_xlabel('Time [s]')
    ax.set_ylabel(ylabs[idx])
    ax.plot(sol.t, costate)

fig_costates.tight_layout()

plt.show()
