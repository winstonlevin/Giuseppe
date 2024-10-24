import pickle
from matplotlib import pyplot as plt

with open('sol_set.data', 'rb') as f:
    sols = pickle.load(f)
    sol = sols[-1]

fig_x_lam, ax_x_lam = plt.subplots(nrows=3, ncols=2, sharex=True)
labs_x = (r'$x$', r'$y$', r'$\psi$')
labs_lam = (r'$\lambda_x$', r'$\lambda_y$', r'$\lambda_\psi$')

for idx, (x, lam) in enumerate(zip(sol.x, sol.lam)):
    ax_x_lam[idx][0].grid()
    ax_x_lam[idx][1].grid()
    ax_x_lam[idx][0].plot(sol.t, x)
    ax_x_lam[idx][1].plot(sol.t, lam)
    ax_x_lam[idx][0].set_ylabel(labs_x[idx])
    ax_x_lam[idx][1].set_ylabel(labs_lam[idx])
    ax_x_lam[idx][0].set_xlabel(r'$t$')
    ax_x_lam[idx][1].set_xlabel(r'$t$')

fig_x_lam.tight_layout()

fig_xy, ax_xy = plt.subplots(nrows=1, ncols=1)
ax_xy.axis('equal')
ax_xy.grid()
ax_xy.set_xlabel(labs_x[0])
ax_xy.set_ylabel(labs_x[1])
ax_xy.plot(sol.x[0, :], sol.x[1, :])
fig_xy.tight_layout()

fig_u, ax_u = plt.subplots(nrows=1, ncols=1)
ax_u.grid()
ax_u.set_xlabel(r'$t$')
ax_u.set_ylabel(r'$u$')
ax_u.plot(sol.t, sol.u[0, :])

plt.show()
