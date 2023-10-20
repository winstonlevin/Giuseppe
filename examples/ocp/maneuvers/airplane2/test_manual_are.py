import numpy as np
import scipy as sp
import casadi as ca

from airplane2_aero_atm import lut_data, CL0, CD1, mass, g0, weight0, s_ref, max_ld_fun

# Flat earth
g = g0

# Exponential atmosphere
rho0 = 0.002378
h_ref = 23_800.
beta1 = 1/h_ref  # -rho_h / rho
beta2 = 1/h_ref**2  # rho_hh / rho

# Constant aero
idx_mach = -1
mach = lut_data['M'][idx_mach]
CLa = lut_data['CLa'][idx_mach]
CD0 = lut_data['CD0'][idx_mach]
CD2 = lut_data['CD2'][idx_mach]
max_ld_dict = max_ld_fun(_CLa=CLa, _CD0=CD0, _CD2=CD2)
max_ld = max_ld_dict['LD']
CL_max_ld = max_ld_dict['CL']
CD_max_ld = max_ld_dict['CD']

# States
E = ca.SX.sym('E', 1)
h = ca.SX.sym('h', 1)
gam = ca.SX.sym('gam', 1)
x_fast = ca.vcat((h, gam))

# Costates
lam_E = ca.SX.sym('lam_E', 1)
lam_h = ca.SX.sym('lam_h', 1)
lam_gam = ca.SX.sym('lam_gam', 1)

# Controls
CL = ca.SX.sym('CL', 1)
u = ca.vcat((CL,))

# Expressions
V = ca.sqrt(2*(E - g*h))
rho = rho0 * ca.exp(-h / h_ref)
qdyn = 0.5 * rho * V**2
lift = qdyn * s_ref * CL
CD = CD0 + CD1 * CL + CD2 * CL**2
drag = qdyn * s_ref * CD

# Dynamics
dtha_dt = V * ca.cos(gam)
dh_dt = V * ca.sin(gam)
dgam_dt = (lift - weight0 * ca.cos(gam))/(mass*V)
dE_dt = - V * drag / mass
f_fast = ca.vcat((dh_dt, dgam_dt))
f_x = ca.jacobian(f_fast, x_fast)
f_u = ca.jacobian(f_fast, u)

# Hamiltonian
hamiltonian = -dtha_dt + lam_E * dE_dt + lam_h * dh_dt + lam_gam * dgam_dt
Hu = ca.jacobian(hamiltonian, u)
Huu = ca.jacobian(Hu, u)
Hx = ca.jacobian(hamiltonian, x_fast)
Hxx = ca.jacobian(Hx, x_fast)
Hxu = ca.jacobian(Hx, u)

# CasADi Functions
Hx_fun_ca = ca.Function(
    'Hx', (E, h, gam, lam_E, lam_h, lam_gam, CL), (Hx,), ('E', 'h', 'gam', 'lam_E', 'lam_h', 'lam_gam', 'CL'), ('Hx',)
)

A_fun_ca = ca.Function(
    'A', (E, h, gam, lam_E, lam_h, lam_gam, CL), (f_x,), ('E', 'h', 'gam', 'lam_E', 'lam_h', 'lam_gam', 'CL'), ('A',)
)
B_fun_ca = ca.Function(
    'B', (E, h, gam, lam_E, lam_h, lam_gam, CL), (f_u,), ('E', 'h', 'gam', 'lam_E', 'lam_h', 'lam_gam', 'CL'), ('B',)
)
Q_fun_ca = ca.Function(
    'Q', (E, h, gam, lam_E, lam_h, lam_gam, CL), (Hxx,), ('E', 'h', 'gam', 'lam_E', 'lam_h', 'lam_gam', 'CL'), ('Q',)
)
N_fun_ca = ca.Function(
    'N', (E, h, gam, lam_E, lam_h, lam_gam, CL), (Hxu,), ('E', 'h', 'gam', 'lam_E', 'lam_h', 'lam_gam', 'CL'), ('N',)
)
R_fun_ca = ca.Function(
    'R', (E, h, gam, lam_E, lam_h, lam_gam, CL), (Huu,), ('E', 'h', 'gam', 'lam_E', 'lam_h', 'lam_gam', 'CL'), ('R',)
)


# Explicit functions
def A_fun(_V):
    _a1 = _V
    _a2 = (beta1 + 2 * g / _V**2) * g / _V
    _A = np.vstack(((0., _a1), (-_a2, 0.)))
    return _A


def B_fun(_V):
    _b = g / (_V * CL_max_ld)
    _B = np.vstack(((0.,), (_b,)))
    return _B


def Q_fun(_V):
    _q = 4 * g/_V * (2 * g/_V**2 + beta1)
    _Q = np.vstack(((_q, 0.), (0., 0.)))
    return _Q


def N_fun(_V):
    _CDu = CD1 + 2 * CD2 * CL_max_ld
    _n = -2 * g/_V * _CDu / CD_max_ld
    _N = np.vstack(((_n,), (0.,)))
    return _N


def R_fun(_V):
    _r = 2*CD2/CD_max_ld * _V
    _R = np.array((_r,)).reshape((1, 1))
    return _R


def check_are(_A, _B, _Q, _N, _R, _P):
    _PBN = _P @ _B + _N
    _ARE = _P @ _A + _A.T @ _P - _PBN @ np.linalg.solve(_R, _PBN.T) + _Q
    return _ARE

# Glide-slope values
h_g = 10e3
rho_g = rho0 * np.exp(-h_g / h_ref)
CL_g = CL_max_ld
CDu_g = CD1 + 2 * CD2 * CL_g
CD_g = CD_max_ld
drag_g = weight0 / max_ld
qdyn_g = weight0 / (s_ref * CL_g)
V2_g = 2 * qdyn_g / rho_g
V_g = np.sqrt(V2_g)
E_g = g * h_g + 0.5 * V2_g

gam_g = 0.
lam_E_g = -mass / drag_g
lam_h_g = 0.
lam_gam_g = V2_g * lam_E_g * CDu_g

A_ca = A_fun_ca(E_g, h_g, gam_g, lam_E_g, lam_h_g, lam_gam_g, CL_g)
B_ca = B_fun_ca(E_g, h_g, gam_g, lam_E_g, lam_h_g, lam_gam_g, CL_g)

Q_ca = Q_fun_ca(E_g, h_g, gam_g, lam_E_g, lam_h_g, lam_gam_g, CL_g)
N_ca = N_fun_ca(E_g, h_g, gam_g, lam_E_g, lam_h_g, lam_gam_g, CL_g)
R_ca = R_fun_ca(E_g, h_g, gam_g, lam_E_g, lam_h_g, lam_gam_g, CL_g)

A = A_fun(V_g)
B = B_fun(V_g)

Q = Q_fun(V_g)
N = N_fun(V_g)
R = R_fun(V_g)

P_sp = sp.linalg.solve_continuous_are(a=A, b=B, q=Q, s=N, r=R)
K_sp = sp.linalg.solve(R, B.T@P_sp + N.T)
err_sp = check_are(A, B, Q, N, R, P_sp)
