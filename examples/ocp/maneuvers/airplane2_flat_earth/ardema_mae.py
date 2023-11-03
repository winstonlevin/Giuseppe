import numpy as np
import scipy as sp
import casadi as ca

from airplane2_aero_atm import g, weight0, dens_fun, sped_fun, s_ref, CD0_fun, CD2_fun, thrust_fun

# This example is Mark Ardema's solution to the Minimum Time to Climb problem by matched asymptotic expansions.
# https://doi.org/10.2514/3.7161

# State Variables
E = ca.MX.sym('E')  # E' / g0**2
h = ca.MX.sym('h')  # h'/g0
gam = ca.MX.sym('gam')  # gam for real :)
v = (2 * (E - h))**0.5  # v' / g0

# Control Variables
L = ca.MX.sym('L')  # L' / W

# "Real" Variables
hp = h * g
vp = v * g
rho = dens_fun(hp)
mach = vp / sped_fun(hp)

# Interpolations
CD0 = CD0_fun(mach)
CD2 = CD2_fun(mach)
thrust = thrust_fun(mach, hp)

# Expressions
Q = 0.5 * rho * vp**2 * s_ref
w = weight0
B = CD2 / Q
D0 = CD0 * Q / w
DL = B * w * L**2
F = thrust/w - D0

# Dynamics
dhdt = v * ca.sin(gam)
dEdt = v * (F - DL)
dgamdt = (L - ca.cos(gam))/v

# Necessary Conditions (MAXIMUM)
lam_E = ca.MX.sym('lam_E')
lam_h = ca.MX.sym('lam_h')
lam_gam = ca.MX.sym('lam_gam')

hamiltonian = -1 + lam_h * dhdt + lam_E * dEdt + lam_gam * dgamdt

dlam_hdt = -ca.jacobian(hamiltonian, h)
dlam_gamdt = -ca.jacobian(hamiltonian, gam)

# Outer solution [Zero function]
gam00 = 0.
L00 = 1.
DL00 = B * w * L00**2
dEdt00 = v * (F - DL00)

lam_h00 = 0.
lam_E00 = 1. / dEdt00
lam_gam00 = v**2 * lam_E00 * (2*B*w*L00)

outer_vars = (lam_gam, lam_E, lam_h, L, gam)
outer_vals = (lam_gam00, lam_E00, lam_h00, L00, gam00)

dlam_hdt00 = dlam_hdt
for outer_var, outer_val in zip(outer_vars, outer_vals):
    dlam_hdt00 = ca.substitute(dlam_hdt00, outer_var, outer_val)


zero_expr = dlam_hdt00
grad_expr = ca.jacobian(zero_expr, h)
zero_fun = ca.Function('F', (E, h), (zero_expr,), ('E', 'h'), ('F',))
grad_fun = ca.Function('DF', (E, h), (grad_expr,), ('E', 'h'), ('DF',))

# Manual Derivation
zero_expr_manual = -lam_E00 * ca.jacobian(dEdt00, h)
grad_expr_manual = ca.jacobian(zero_expr, h)
zero_fun_manual = ca.Function('F', (E, h), (zero_expr_manual,), ('E', 'h'), ('F',))
grad_fun_manual = ca.Function('DF', (E, h), (grad_expr_manual,), ('E', 'h'), ('DF',))

# Fast dynamics
z_vec = ca.vcat((h, gam, lam_h, lam_gam))
dzdt = ca.vcat((dhdt, dgamdt, dlam_hdt, dlam_gamdt))
L_cl = lam_gam / (v**2 * lam_E * (2*B*w))
dzdt_cl = ca.substitute(dzdt, L, L_cl)
G = ca.jacobian(dzdt_cl, z_vec)
G_fun = ca.Function(
    'G', (E, h, gam, lam_E, lam_h, lam_gam), (G,), ('E', 'h', 'gam', 'lam_E', 'lam_h', 'lam_gam'), ('G',)
)

# Energy State Solution
E_es = ca.MX.sym('E')
h_es = ca.MX.sym('h')
v_es = (2 * (E_es - g * h_es))**0.5
mach_es = v_es / sped_fun(h_es)
thrust_es = thrust_fun(mach_es, h_es)
lift_es = w
qdyn_s_ref_es = dens_fun(h_es) * (E_es - g * h_es) * s_ref
drag_es = CD0_fun(mach_es) * qdyn_s_ref_es + CD2_fun(mach_es) / qdyn_s_ref_es * lift_es**2

obj_es = v_es * (thrust_es - drag_es) / weight0
zero_es = ca.jacobian(obj_es, h_es)
grad_es = ca.jacobian(zero_es, h_es)

obj_fun_es = ca.Function('F', (E_es, h_es), (obj_es,), ('E', 'h'), ('F',))
zero_fun_es = ca.Function('Fz', (E_es, h_es), (zero_es,), ('E', 'h'), ('Fz',))
grad_fun_es = ca.Function('DFz', (E_es, h_es), (grad_es,), ('E', 'h'), ('DFz',))

# NUMERICAL SOLUTION [from ref] ----------------------------------------------------------------------------------------
r2d = 180 / np.pi

hp0 = 40e3
machp0 = 0.5
vp0 = machp0 * float(sped_fun(hp0))
h0 = hp0/g
v0 = vp0/g
E0 = h0 + 0.5 * v0**2
Ep0 = g * hp0 + 0.5 * vp0**2

# Outer Solution
h00 = sp.optimize.fsolve(
    func=lambda _x: np.asarray(zero_fun(E0, _x[0])).flatten(),
    x0=np.array((7.5e3 / g,)),
    fprime=lambda _x:  np.asarray(grad_fun(E0, _x[0]))
)[0]
v00 = (2 * (E0 - h00))**0.5  # v' / g0
hp00 = h00 * g
vp00 = v00 * g
mach00 = vp00 / float(sped_fun(hp00))
CD000 = float(CD0_fun(mach00))
CD200 = float(CD2_fun(mach00))
thrust00 = float(thrust_fun(mach00, hp00))

rho00 = dens_fun(hp00)
Q00 = 0.5 * rho00 * vp00**2 * s_ref
B00 = CD200 / Q00
F0 = thrust00/w - CD000

D000 = CD000 * Q00 / w
DL00 = B00 * w * L00**2
F00 = thrust00/w - D000
dEdt00 = v00 * (F00 - DL00)

lam_h00 = 0.
lam_E00 = 1. / dEdt00
lam_gam00 = 2 * v00**2 * lam_E00 * B00 * w

# Linearization about E
G00 = np.asarray(G_fun(E0, h00, gam00, lam_E00, lam_h00, lam_gam00))
eig_G00, eig_vec_G00 = np.linalg.eig(G00)
idx_stable = np.where(np.real(eig_G00) < 0)
idx_unstable = np.where(np.real(eig_G00) > 0)
eig_vec_stable = eig_vec_G00[:, idx_stable[0]]
eig_vec_unstable = eig_vec_G00[:, idx_unstable[0]]
Vs = np.hstack((np.real(eig_vec_stable[:, 0].reshape(-1, 1)), np.imag(eig_vec_stable[:, 0].reshape(-1, 1))))
Vu = np.hstack((np.real(eig_vec_unstable[:, 0].reshape(-1, 1)), np.imag(eig_vec_unstable[:, 0].reshape(-1, 1))))
Vs_known = Vs[(0, 3), :]  # h0 fixed, gam0 free -> lam_gam0 = 0
Vs_unknown = Vs[(1, 2), :]

# Solve for unknown initial conditions
dh0 = h0 - h00
dlam_gam0 = 0.
z0_known = np.vstack((dh0, dlam_gam0))
z0_unknown = Vs_unknown @ np.linalg.solve(Vs_known, z0_known)
dgam0 = z0_unknown[0, 0]
dlam_h0 = z0_unknown[1, 0]

z0 = np.array((dh0, dgam0, dlam_h0, dgam0))
sol_ivp = sp.integrate.solve_ivp(fun=lambda t, z: (G00 @ z).flatten(), t_span=np.array((0., 60.)), y0=z0)

from matplotlib import pyplot as plt

fig = plt.figure()

ax1 = fig.add_subplot(211)
ax1.plot(sol_ivp.t, sol_ivp.y[0, :] * g)
ax1.grid()
ax1.set_xlabel('t [s]')
ax1.set_ylabel('dh [ft]')

ax2 = fig.add_subplot(212)
ax2.plot(sol_ivp.t, sol_ivp.y[1, :] * r2d)
ax2.grid()
ax2.set_xlabel('t [s]')
ax2.set_ylabel('dgam [deg]')

fig.tight_layout()

plt.show()
