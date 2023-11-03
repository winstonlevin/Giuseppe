import numpy as np
import scipy as sp
import casadi as ca

from airplane2_aero_atm import g, weight0, dens_fun, sped_fun, s_ref, CD0_fun, CD2_fun, thrust_fun

# This example is Mark Ardema's solution to the Minimum Time to Climb problem by matched asymptotic expansions.
# https://doi.org/10.2514/3.7161

# State Variables
E = ca.MX.sym('E')
h = ca.MX.sym('h')
gam = ca.MX.sym('gam')
m = ca.MX.sym('m')
v = (2 * (E - g*h))**0.5

# Control Variables
lift = ca.MX.sym('lift')

# Expressions
rho = dens_fun(h)
mach = v / sped_fun(h)
qdyn = 0.5 * rho * v**2
CD0 = CD0_fun(mach)
CD2 = CD2_fun(mach)
thrust = thrust_fun(mach, h)
drag = CD0 * qdyn * s_ref + CD2/(qdyn * s_ref) * lift**2
weight = m * g

# Dynamics
dhdt = v * ca.sin(gam)
dEdt = v * (thrust - drag) / m
dgamdt = (lift - weight * ca.cos(gam))/(m*v)

# Necessary Conditions (MAXIMUM)
lam_E = ca.MX.sym('lam_E')
lam_h = ca.MX.sym('lam_h')
lam_gam = ca.MX.sym('lam_gam')

hamiltonian = 1 + lam_h * dhdt + lam_E * dEdt + lam_gam * dgamdt

dlam_hdt = -ca.jacobian(hamiltonian, h)
dlam_gamdt = -ca.jacobian(hamiltonian, gam)

# Fast dynamics
z_vec = ca.vcat((h, gam, lam_h, lam_gam))
dzdt = ca.vcat((dhdt, dgamdt, dlam_hdt, dlam_gamdt))
L_cl = lam_gam/lam_E * rho * s_ref / (4 * CD2)
dzdt_cl = ca.substitute(dzdt, lift, L_cl)
G = ca.jacobian(dzdt_cl, z_vec)
G_fun = ca.Function(
    'G', (m, E, h, gam, lam_E, lam_h, lam_gam), (G,), ('m', 'E', 'h', 'gam', 'lam_E', 'lam_h', 'lam_gam'), ('G',)
)
L_cl_fun = ca.Function(
    'L', (m, E, h, gam, lam_E, lam_h, lam_gam), (L_cl,), ('m', 'E', 'h', 'gam', 'lam_E', 'lam_h', 'lam_gam'), ('G',)
)

# Energy State Solution
drag_es = ca.substitute(drag, lift, weight)

obj_es = v * (thrust - drag_es) / weight
zero_es = ca.jacobian(obj_es, h)
grad_es = ca.jacobian(zero_es, h)

obj_fun_es = ca.Function('F', (m, E, h), (obj_es,), ('m', 'E', 'h'), ('F',))
zero_fun_es = ca.Function('Fz', (m, E, h), (zero_es,), ('m', 'E', 'h'), ('Fz',))
grad_fun_es = ca.Function('DFz', (m, E, h), (grad_es,), ('m', 'E', 'h'), ('DFz',))


def find_climb_path(mass, energy, h_guess):
    _h = sp.optimize.fsolve(
        func=lambda _x: np.asarray(zero_fun_es(mass, energy, _x[0])).flatten(),
        x0=np.array((h_guess,)),
        fprime=lambda _x: np.asarray(grad_fun_es(mass, energy, _x[0]))
    )[0]
    _v = (2 * (energy - g * _h))**0.5
    _mach = _v / float(sped_fun(_h))
    _qdyn_s_ref = 0.5 * float(dens_fun(_h)) * _v**2 * s_ref
    _thrust = thrust_fun(_mach, _h)
    _CD0 = float(CD0_fun(_mach))
    _CD2 = float(CD2_fun(_mach))
    _lift = g * mass
    _drag = _qdyn_s_ref * _CD0 + _CD2 / _qdyn_s_ref * _lift**2
    _DL = 2 * _CD2/_qdyn_s_ref * _lift
    _dEdt = _v * (_thrust - _drag) / mass
    _lam_E = -1 / _dEdt
    _lam_gam = _lam_E * _v**2 * _DL
    _lam_h = 0.
    _gam = 0.
    _climb_dict = {
        'm': mass, 'E': energy, 'h': _h, 'V': _v, 'gam': _gam,
        'lam_E': _lam_E, 'lam_h': _lam_h, 'lam_gam': _lam_gam,
        'L': _lift, 'D': _drag, 'T': _thrust, 'M': _mach, 'qdyn_s_ref': _qdyn_s_ref
    }
    return _climb_dict


# NUMERICAL SOLUTION [from ref] ----------------------------------------------------------------------------------------
r2d = 180 / np.pi

mass0 = weight0 / g
h0 = 40e3
mach0 = 0.5
v0 = mach0 * float(sped_fun(h0))
E0 = g * h0 + 0.5 * v0**2

# Outer Solution
_outer_dict = find_climb_path(mass0, E0, 8.e3)

# Linearization about E
G00 = np.asarray(G_fun(
    mass0, E0, _outer_dict['h'], _outer_dict['gam'], _outer_dict['lam_E'], _outer_dict['lam_h'], _outer_dict['lam_gam']
))
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
dh0 = h0 - _outer_dict['h']
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
