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
x_feedback = ca.vcat((v, gam))

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
f_feedback = ca.vcat((dv_dt, dgam_dt))

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

# Convert Expressions to Functions for LQR
A_fun = ca.Function('A', (x, lam, u), (A,), ('x', 'lam', 'u'), ('A',))
B_fun = ca.Function('B', (x, lam, u), (B,), ('x', 'lam', 'u'), ('B',))
Q_fun = ca.Function('Q', (x, lam, u), (Q,), ('x', 'lam', 'u'), ('Q',))
R_fun = ca.Function('Q', (x, lam, u), (R,), ('x', 'lam', 'u'), ('R',))
N_fun = ca.Function('Q', (x, lam, u), (N,), ('x', 'lam', 'u'), ('N',))

# Generate Interpolators for K based on E ------------------------------------------------------------------------------
e_vals = np.empty(sol.t.shape)

alp_glide_vals = np.empty(sol.t.shape)
v_glide_vals = np.empty(sol.t.shape)
gam_glide_vals = np.empty(sol.t.shape)

k_v_vals = np.empty(sol.t.shape)
k_gam_vals = np.empty(sol.t.shape)


for idx in range(len(sol.t)):
    # Start at last idx (minimum energy) for monotonically increasing sequence.
    idx_rev = -1 - idx
    h_idx = x_dict['h'][idx_rev]
    xn_idx = x_dict['xn'][idx_rev]
    v_idx = x_dict['v'][idx_rev]
    gam_idx = x_dict['gam'][idx_rev]
    x_idx = np.vstack((h_idx, xn_idx, v_idx, gam_idx))

    g_idx = mu / (Re + h_idx) ** 2
    e_idx = g_idx * h_idx + 0.5 * v_idx ** 2

    lam_h_idx = lam_dict['h'][idx_rev] * k_dict['h_ref']
    lam_xn_idx = lam_dict['xn'][idx_rev] * k_dict['x_ref']
    lam_v_idx = lam_dict['v'][idx_rev] * k_dict['v_ref']
    lam_gam_idx = lam_dict['gam'][idx_rev] * k_dict['gam_ref']
    lam_idx = np.vstack((lam_h_idx, lam_xn_idx, lam_v_idx, lam_gam_idx))

    alp_idx = u_dict['alpha'][idx_rev]
    u_idx = np.vstack((alp_idx,))

    A_idx = np.asarray(A_fun(x_idx, lam_idx, u_idx))
    B_idx = np.asarray(B_fun(x_idx, lam_idx, u_idx))
    Q_idx = np.asarray(Q_fun(x_idx, lam_idx, u_idx))
    R_idx = np.asarray(R_fun(x_idx, lam_idx, u_idx))
    N_idx = np.asarray(N_fun(x_idx, lam_idx, u_idx))

    P_idx = sp.linalg.solve_continuous_are(A_idx, B_idx, Q_idx, R_idx, s=N_idx)
    K_idx = sp.linalg.solve(R_idx, B_idx.T @ P_idx + N_idx.T)

    e_vals[idx] = e_idx

    v_glide_vals[idx] = v_idx
    gam_glide_vals[idx] = gam_idx
    alp_glide_vals[idx] = alp_idx

    k_v_vals[idx] = K_idx[0, 0]
    k_gam_vals[idx] = K_idx[0, 1]

v_glide_interp = sp.interpolate.pchip(e_vals, v_glide_vals)
gam_glide_interp = sp.interpolate.pchip(e_vals, gam_glide_vals)
alp_glide_interp = sp.interpolate.pchip(e_vals, alp_glide_vals)

k_v_interp = sp.interpolate.pchip(e_vals, k_v_vals)
k_gam_interp = sp.interpolate.pchip(e_vals, k_gam_vals)

interp_dict = {
    'v': v_glide_interp,
    'gam': gam_glide_interp,
    'alp': alp_glide_interp,
    'k_v': k_v_interp,
    'k_gam': k_gam_interp
}

# Control Law: alp = alp(e) - Kv(e) * (V - V(e)) - Kgam(e) * (gam - gam(e))

with open('neighboring_feedback_dict_range.data', 'wb') as file:
    pickle.dump(interp_dict, file)
