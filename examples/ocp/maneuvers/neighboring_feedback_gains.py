import pickle
import scipy as sp
import numpy as np
import casadi as ca

from lookup_tables import thrust_table, cl_alpha_table, cd0_table, temp_table, sped_table, dens_table, atm

# Load Optimal Glide Path ----------------------------------------------------------------------------------------------
with open('sol_set_range.data', 'rb') as f:
    sols = pickle.load(f)
    sol = sols[-1]

# Generate dictionaries
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

# Construct simplified feedback state (h, V, gam) ----------------------------------------------------------------------
idx_feedback = (0, 2, 3)

# States
h = ca.MX.sym('h')
xn = ca.MX.sym('xn')
v = ca.MX.sym('v')
gam = ca.MX.sym('gam')

x_tuple = (h, xn, v, gam)
x = ca.vcat(x_tuple)
x_feedback = ca.vcat([])
for idx in idx_feedback:
    x_feedback = ca.vcat((x_feedback, x_tuple[idx]))

# Costates
lam_h = ca.MX.sym('lam_h')
lam_xn = ca.MX.sym('lam_xn')
lam_v = ca.MX.sym('lam_v')
lam_gam = ca.MX.sym('lam_gam')
lam = ca.vcat((lam_h, lam_xn, lam_v, lam_gam))

# Control
alpha = ca.MX.sym('alpha', 1)
u = alpha
eps_u = ca.MX.sym('eps_h', 1)

# Constants
s_ref = k_dict['s_ref']
eta = k_dict['eta']
mu = k_dict['mu']
Re = k_dict['Re']
m = x_dict['m'][0]

# Atmosphere and Aerodynamics
a = sped_table(h)
mach = v / a
rho = dens_table(h)
qdyn = 0.5 * rho * v**2
g = mu / (Re + h) ** 2

cla = cl_alpha_table(mach)
cd0 = cd0_table(mach)
lift = qdyn * s_ref * cla * alpha
drag = qdyn * s_ref * (cd0 + eta * cla * alpha ** 2)

# Equations of Motion
dh_dt = v * ca.sin(gam)
dxn_dt = v * ca.cos(gam)
dv_dt = drag / m - g * ca.sin(gam)
dgam_dt = lift / (m * v) - g / v * ca.cos(gam)

f_tuple = (dh_dt, dxn_dt, dv_dt, dgam_dt)
f = ca.vcat(f_tuple)
f_feedback = ca.vcat([])
for idx in idx_feedback:
    f_feedback = ca.vcat((f_feedback, f_tuple[idx]))

# Linearization of EoM
A = ca.jacobian(f_feedback, x_feedback)
B = ca.jacobian(f_feedback, u)

# Hamiltonian and Partial Derivatives
ham = lam.T @ f + eps_u * alpha ** 2

lam_dot = -ca.jacobian(ham, x).T
Hx = ca.jacobian(ham, x_feedback)
Hu = ca.jacobian(ham, u)
Hxx = ca.jacobian(Hx, x_feedback)
Hxu = ca.jacobian(Hx, u)
Huu = ca.jacobian(Hu, u)

Q = Hxx
R = Huu
N = Hxu

Rinv_NT = ca.solve(R, N.T)
Rinv_BT = ca.solve(R, B.T)
Q_aug = Q - N @ Rinv_NT
A_aug = A - B @ Rinv_NT

P = ca.MX.sym('P', A.shape)
dP_dt = -(Q_aug + P @ A_aug + A_aug.T @ P - P @ B @ Rinv_BT @ P)
# dP_dt = -(A.T @ P + P @ A - (P @ B + N) @ ca.solve(R, B.T @ P + N.T) + Q)
K = ca.solve(R, B.T @ P + N.T)

# Optimal Control Law (Analytical Solution to Hu = 0)
alpha_opt = lam_gam / (lam_v * 2 * eta * v)

# Convert Expressions to Functions for Riccati Differential Equation
ctrl_law = ca.Function('u', (x, lam), (alpha_opt,), ('x', 'lam'), ('alpha',))
dx_dt = ca.Function('f', (x, u), (f,), ('x', 'u'), ('dx_dt',))
dlam_dt = ca.Function('flam', (x, lam, u), (lam_dot,), ('x', 'lam', 'u'), ('dlam_dt',))

A_fun = ca.Function('A', (x, lam, u), (A,), ('x', 'lam', 'u'), ('A',))
B_fun = ca.Function('B', (x, lam, u), (B,), ('x', 'lam', 'u'), ('B',))
Q_fun = ca.Function('Q', (x, lam, u, eps_u), (Q,), ('x', 'lam', 'u', 'eps_u'), ('Q',))
R_fun = ca.Function('R', (x, lam, u, eps_u), (R,), ('x', 'lam', 'u', 'eps_u'), ('R',))
N_fun = ca.Function('N', (x, lam, u, eps_u), (N,), ('x', 'lam', 'u', 'eps_u'), ('N',))

dP_dt_fun = ca.Function('dPdt', (P, x, lam, u, eps_u), (dP_dt,), ('P', 'x', 'lam', 'u', 'eps_u'), ('dPdt',))
K_fun = ca.Function('K', (P, x, lam, u, eps_u), (K,), ('P', 'x', 'lam', 'u', 'eps_u'), ('K',))

# Generate Interpolators for K based on E ------------------------------------------------------------------------------

shape_P = P.shape
idx_lam0 = x.numel()
idx_p0 = 2 * x.numel()


def dx_lam_p_dt(_t, _x_lam_p, _eps_u):
    # These equations of motion are to integrate p from tf (tgo = 0) to t0 (tgo = tf).
    # The Riccati differentiation equation is:
    # dP/dtgo = -dP/dt = A'P + PA - (PB + N) R^-1 (B'P + N') + Q
    _x = _x_lam_p[0:idx_lam0]
    _lam = _x_lam_p[idx_lam0:idx_p0]
    _p = _x_lam_p[idx_p0:].reshape(shape_P)
    _u = np.asarray(ctrl_law(_x, _lam))

    _dx_dt = np.asarray(dx_dt(_x, _u)).flatten()
    _dlam_dt = np.asarray(dlam_dt(_x, _lam, _u)).flatten()
    _dp_dt = np.asarray(dP_dt_fun(_p, _x, _lam, _u, _eps_u)).flatten()

    # if np.max(np.abs(_dp_dt)) > 1e7:
    #     print(f'dP/dt(t = {_t}):')
    #     print(_dp_dt)
    #     print(f'\n')

    return np.concatenate((_dx_dt, _dlam_dt, _dp_dt))


# Start at final time with known P(tf) and integrate backwards
# Initialize with P(tf) = 0 [Since Phi_xx = 0]
P_idx = np.zeros(shape_P)
P_vec_idx = P_idx.flatten()

idx_t0 = 0
idx_tf = len(sol.t) - 1

# t_start = 550
# t_end = 650
# t_idces = np.where(np.logical_and(sol.t > t_start, sol.t < t_end))
# idx_t0 = t_idces[0][0]
# idx_tf = t_idces[0][-1]

num_t = 1 + idx_tf - idx_t0

idx_tgo = idx_tf - (idx_t0 + 0)
xf = np.array((x_dict['h'][idx_tgo],
               x_dict['xn'][idx_tgo],
               x_dict['v'][idx_tgo],
               x_dict['gam'][idx_tgo]))

lamf = np.array((lam_dict['h'][idx_tgo],
                 lam_dict['xn'][idx_tgo],
                 lam_dict['v'][idx_tgo],
                 lam_dict['gam'][idx_tgo]))

K_mat = np.empty((x_feedback.numel(), num_t))
eps_u_idx = 0

for idx in range(num_t):
    idx_tgo = idx_tf - (idx_t0 + idx)
    x_idx = np.array((x_dict['h'][idx_tgo],
                      x_dict['xn'][idx_tgo],
                      x_dict['v'][idx_tgo],
                      x_dict['gam'][idx_tgo]))
    lam_idx = np.array((lam_dict['h'][idx_tgo],
                        lam_dict['xn'][idx_tgo],
                        lam_dict['v'][idx_tgo],
                        lam_dict['gam'][idx_tgo]))

    print(f'SSE(x - xf) = {np.sum((x_idx - xf)**2)}')
    print(f'SSE(lam - lamf) = {np.sum((lam_idx - lamf) ** 2)}')
    # lam_idx = np.array((lam_dict['h'][idx_tgo] * k_dict['h_ref'],
    #                     lam_dict['xn'][idx_tgo] * k_dict['x_ref'],
    #                     lam_dict['v'][idx_tgo] * k_dict['v_ref'],
    #                     lam_dict['gam'][idx_tgo] * k_dict['gam_ref']))
    y_idx = np.concatenate((x_idx, lam_idx, P_vec_idx))
    u_idx = np.array((u_dict['alpha'][idx_tgo],))
    K_mat[:, idx_tgo] = np.asarray(K_fun(P_idx, x_idx, lam_idx, u_idx, eps_u_idx))

    tspan = (sol.t[idx_tgo], sol.t[idx_tgo - 1])

    ivp_sol = sp.integrate.solve_ivp(lambda _t, _x_lam_p: dx_lam_p_dt(_t, _x_lam_p, eps_u_idx), tspan, y_idx)

    xf = ivp_sol.y[:idx_lam0, -1]
    lamf = ivp_sol.y[idx_lam0:idx_p0, -1]
    Pf = ivp_sol.y[idx_p0:, -1].reshape(shape_P)
    uf = ctrl_law(xf, lamf)
    Af = A_fun(xf, lamf, uf)
    Bf = B_fun(xf, lamf, uf)
    Qf = Q_fun(xf, lamf, uf, eps_u_idx)
    Rf = R_fun(xf, lamf, uf, eps_u_idx)
    Nf = N_fun(xf, lamf, uf, eps_u_idx)
    dPf = dP_dt_fun(Pf, xf, lamf, uf, eps_u_idx)

    # print(f'flam_min[Q(t={sol.t[idx_tgo]})]:')
    # print(np.min(np.linalg.eigvals(Qf)))

    if not ivp_sol.success:
        print('Integration Failed!')
        break

    P_vec_idx = ivp_sol.y[idx_p0:, -1]
    P_idx = P_vec_idx.reshape(shape_P)

# for idx, (tgo, p_flat) in enumerate(zip(ivp_sol.t, ivp_sol.y.T)):
#     P_idx = p_flat.reshape(_shape_P)
#     x_idx, lam_idx, u_idx = get_state(tgo)
#     K_idx = np.asarray(K_fun(P_idx, x_idx, lam_idx, u_idx))

# k_v_vals = np.empty(sol.t.shape)
# k_gam_vals = np.empty(sol.t.shape)

# v_glide_interp = sp.interpolate.pchip(e_vals, v_glide_vals)
# gam_glide_interp = sp.interpolate.pchip(e_vals, gam_glide_vals)
# alp_glide_interp = sp.interpolate.pchip(e_vals, alp_glide_vals)
#
# k_v_interp = sp.interpolate.pchip(e_vals, k_v_vals)
# k_gam_interp = sp.interpolate.pchip(e_vals, k_gam_vals)
#
# interp_dict = {
#     'v': v_glide_interp,
#     'gam': gam_glide_interp,
#     'alp': alp_glide_interp,
#     'k_v': k_v_interp,
#     'k_gam': k_gam_interp
# }
#
# # Control Law: alp = alp(e) - Kv(e) * (V - V(e)) - Kgam(e) * (gam - gam(e))
#
# with open('neighboring_feedback_dict_range.data', 'wb') as file:
#     pickle.dump(interp_dict, file)
