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

# States
h = ca.MX.sym('h')
xn = ca.MX.sym('xn')
v = ca.MX.sym('v')
gam = ca.MX.sym('gam')

x = ca.vcat((h, xn, v, gam))
x_feedback = ca.vcat((h, v, gam))

# Costates
lam_h = ca.MX.sym('lam_h')
lam_xn = ca.MX.sym('lam_xn')
lam_v = ca.MX.sym('lam_v')
lam_gam = ca.MX.sym('lam_gam')
lam = ca.vcat((lam_h, lam_xn, lam_v, lam_gam))

# Control
alpha = ca.MX.sym('alpha', 1)
u = alpha

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

f = ca.vcat((dh_dt, dxn_dt, dv_dt, dgam_dt))
f_feedback = ca.vcat((dh_dt, dv_dt, dgam_dt))

# Linearization of EoM
A = ca.jacobian(f_feedback, x_feedback)
B = ca.jacobian(f_feedback, u)

# Hamiltonian and Partial Derivatives
ham = lam.T @ f

Hx = ca.jacobian(ham, x_feedback)
Hu = ca.jacobian(ham, u)
Hxx = ca.jacobian(Hx, x_feedback)
Hxu = ca.jacobian(Hx, u)
Huu = ca.jacobian(Hu, u)

Q = Hxx
R = Huu
N = Hxu

P = ca.MX.sym('P', A.shape)
dPdtgo = A.T @ P + P @ A - (P @ B + N) @ ca.solve(R, B.T @ P + N.T) + Q  # -dP/dt
K = ca.solve(R, B.T @ P + N.T)

# Convert Expressions to Functions for Riccati Differential Equation
dPdtgo_fun = ca.Function('dPdtgo', (P, x, lam, u), (dPdtgo,), ('P', 'x', 'lam', 'u'), ('dPdtgo',))
K_fun = ca.Function('K', (P, x, lam, u), (K,), ('P', 'x', 'lam', 'u'), ('K',))

# Generate Interpolators for K based on E ------------------------------------------------------------------------------

# Interpolate the state based on time-to-go tgo
tf = sol.t[-1]
tgo = tf - sol.t

h_interp_tgo = sp.interpolate.pchip(np.flip(tgo), np.flip(x_dict['h']))
xn_interp_tgo = sp.interpolate.pchip(np.flip(tgo), np.flip(x_dict['xn']))
v_interp_tgo = sp.interpolate.pchip(np.flip(tgo), np.flip(x_dict['v']))
gam_interp_tgo = sp.interpolate.pchip(np.flip(tgo), np.flip(x_dict['gam']))

lam_h_interp_tgo = sp.interpolate.pchip(np.flip(tgo), np.flip(lam_dict['h'] * k_dict['h_ref']))
lam_xn_interp_tgo = sp.interpolate.pchip(np.flip(tgo), np.flip(lam_dict['xn'] * k_dict['x_ref']))
lam_v_interp_tgo = sp.interpolate.pchip(np.flip(tgo), np.flip(lam_dict['v'] * k_dict['v_ref']))
lam_gam_interp_tgo = sp.interpolate.pchip(np.flip(tgo), np.flip(lam_dict['gam'] * k_dict['gam_ref']))

alpha_interp_tgo = sp.interpolate.pchip(np.flip(tgo), np.flip(u_dict['alpha']))

g = k_dict['mu'] / (k_dict['Re'] + x_dict['h']) ** 2
e = x_dict['h'] * g + 0.5 * x_dict['v'] ** 2
ego = np.flip(e)

_shape_P = P.shape


def get_state(_tgo):
    _x = np.vstack((h_interp_tgo(_tgo), xn_interp_tgo(_tgo), v_interp_tgo(_tgo), gam_interp_tgo(_tgo)))
    _lam = np.vstack((lam_h_interp_tgo(_tgo), lam_xn_interp_tgo(_tgo), lam_v_interp_tgo(_tgo), lam_gam_interp_tgo(_tgo)))
    _u = np.vstack((alpha_interp_tgo(_tgo),))
    return _x, _lam, _u


def dp_dtgo(_tgo, _p_flat):
    # These equations of motion are to integrate p from tf (tgo = 0) to t0 (tgo = tf).
    # The Riccati differentiation equation is:
    # dP/dtgo = -dP/dt = A'P + PA - (PB + N) R^-1 (B'P + N') + Q
    _p = _p_flat.reshape(_shape_P)
    _x, _lam, _u = get_state(_tgo)
    _dp_dtgo = np.asarray(dPdtgo_fun(_p, _x, _lam, _u))
    return _dp_dtgo.reshape((-1,))


# TODO - fix "Required step size is less than spacing between numbers"
P0 = np.zeros(_shape_P)
ivp_sol = sp.integrate.solve_ivp(dp_dtgo, sol.t[((0, -1),)], P0.flatten(), t_eval=np.flip(tgo))

for idx, (tgo, p_flat) in enumerate(zip(ivp_sol.t, ivp_sol.y)):
    P_idx = p_flat.reshape(_shape_P)
    x_idx, lam_idx, u_idx = get_state(tgo)
    K_idx = np.asarray(K_fun(P_idx, x_idx, lam_idx, u_idx))

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
