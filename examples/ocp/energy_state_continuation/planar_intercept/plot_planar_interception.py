import pickle

import matplotlib as mpl
from matplotlib import pyplot as plt

mpl.rcParams['axes.formatter.useoffset'] = False

FILE_NAME = 'sol_set'

with open(FILE_NAME + '.data', 'rb') as f:
    sols = pickle.load(f)
    sol = sols[-1]

with open('sol_nlp.data', 'rb') as f:
    sol_nlp = pickle.load(f)

# Plotting ----------------------------------------------------------------------------------------------------------- #
linestyle_bvp = '*b'
markersize_bvp = 3
linestyle_nlp = 'or'
markersize_nlp = 2

fig_x_lam, ax_x_lam = plt.subplots(nrows=3, ncols=2, sharex=True)
labs_x = (r'$x$', r'$y$', r'$\psi$')
labs_lam = (r'$\lambda_x$', r'$\lambda_y$', r'$\lambda_\psi$')

for idx, (x, lam) in enumerate(zip(sol.x, sol.lam)):
    ax_x_lam[idx][0].grid()
    ax_x_lam[idx][1].grid()
    ax_x_lam[idx][0].plot(sol.t, x, linestyle_bvp, markersize=markersize_bvp, label='BVP')
    ax_x_lam[idx][1].plot(sol.t, lam, linestyle_bvp, markersize=markersize_bvp, label='BVP')
    ax_x_lam[idx][0].plot(sol_nlp.t, sol_nlp.x[idx, :], linestyle_nlp, markersize=markersize_nlp, label='NLP')
    ax_x_lam[idx][1].plot(sol_nlp.t, sol_nlp.lam[idx, :], linestyle_nlp, markersize=markersize_nlp, label='NLP')
    ax_x_lam[idx][0].set_ylabel(labs_x[idx])
    ax_x_lam[idx][1].set_ylabel(labs_lam[idx])
    ax_x_lam[idx][0].set_xlabel(r'$t$')
    ax_x_lam[idx][1].set_xlabel(r'$t$')
ax_x_lam[0][0].legend()
fig_x_lam.tight_layout()

fig_xy, ax_xy = plt.subplots(nrows=1, ncols=1)
ax_xy.axis('equal')
ax_xy.grid()
ax_xy.set_xlabel(labs_x[0])
ax_xy.set_ylabel(labs_x[1])
ax_xy.plot(sol.x[0, :], sol.x[1, :], linestyle_bvp, markersize=markersize_bvp, label='BVP')
ax_xy.plot(sol_nlp.x[0, :], sol_nlp.x[1, :], linestyle_nlp, markersize=markersize_nlp, label='NLP')
fig_xy.tight_layout()

fig_u, ax_u = plt.subplots(nrows=1, ncols=1)
ax_u.grid()
ax_u.set_xlabel(r'$t$')
ax_u.set_ylabel(r'$u$')
ax_u.plot(sol.t, sol.u[0, :], linestyle_bvp, markersize=markersize_bvp, label='BVP')
ax_u.plot(sol_nlp.t, sol_nlp.u[0, :], linestyle_nlp, markersize=markersize_nlp, label='NLP (u)')
# ax_u.plot(sol_nlp.t, -sol_nlp.lam[2, :]/sol_nlp.find('k'), '*:', label='NLP (-lamPsi/k)')
ax_u.legend()
fig_u.tight_layout()

plt.show()
