import pickle
from matplotlib import pyplot as plt

with open('sol_set.data', 'rb') as f:
    sols = pickle.load(f)
    sol = sols[-1]

fig = plt.figure()

ax_y = fig.add_subplot(211)
ax_y.grid()
ax_y.plot(sol.t, sol.x[0, :])
ax_y.set_ylabel('y')

ax_u = fig.add_subplot(212)
ax_u.grid()
ax_u.plot(sol.t, sol.u[0, :])
ax_u.set_ylabel('u')

fig.tight_layout()

plt.show()
