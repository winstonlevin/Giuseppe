from typing import Callable

import pickle
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

from lookup_tables import cl_alpha_table, cd0_table, thrust_table, atm

mpl.rcParams['axes.formatter.useoffset'] = False
col = plt.rcParams['axes.prop_cycle'].by_key()['color']

COMPARISON = 'max_range'
AOA_LAW = 'weight'  # {weight, max_ld, 0}
ROLL_LAW = '0'  # {0}
THRUST_LAW = '0'  # {0, min, max}

if COMPARISON == 'max_range':
    with open('sol_set_range.data', 'rb') as f:
        sols = pickle.load(f)
        sol = sols[-1]

# Create Dicts
k_dict = {}
p_dict = {}
x_dict = {}
u_dict = {}

for key, val in zip(sol.annotations.constants, sol.k):
    k_dict[key] = val
for key, val in zip(sol.annotations.parameters, sol.p):
    p_dict[key] = val
for key, val in zip(sol.annotations.states, list(sol.x)):
    x_dict[key] = val
for key, val in zip(sol.annotations.controls, list(sol.u)):
    u_dict[key] = val


def generate_constant_ctrl(_const: float) -> Callable:
    def _const_control(_t: float, _x: np.array, _u: np.array, _p_dict: dict, _k_dict: dict) -> float:
        return _const
    return _const_control


def alpha_max_ld(_t: float, _x: np.array, _u: np.array, _p_dict: dict, _k_dict: dict) -> float:
    _mach = _x[3] / atm.speed_of_sound(_x[0])
    return float(cd0_table(_mach) / (_k_dict['eta'] * cl_alpha_table(_mach))) ** 0.5


def alpha_weight(_t: float, _x: np.array, _u: np.array, _p_dict: dict, _k_dict: dict) -> float:
    _mach = _x[3] / atm.speed_of_sound(_x[0])
    _qdyn = 0.5 * atm.density(_x[0]) * _x[3] ** 2
    _weight = _x[6] * _k_dict['mu'] / (_x[0] + _k_dict['Re']) ** 2
    return _weight / float(_qdyn * _k_dict['s_ref'] * cl_alpha_table(_mach))


def generate_ctrl_law() -> Callable:
    if AOA_LAW == 'max_ld':
        _aoa_ctrl = alpha_max_ld
    elif AOA_LAW == 'weight':
        _aoa_ctrl = alpha_weight
    else:
        _aoa_ctrl = generate_constant_ctrl(0.)

    if ROLL_LAW == '0':
        _roll_ctrl = generate_constant_ctrl(0.)

    if THRUST_LAW == '0':
        _thrust_ctrl = generate_constant_ctrl(0.)
    elif THRUST_LAW == 'min':
        _thrust_ctrl = generate_constant_ctrl(0.3)
    elif THRUST_LAW == 'max':
        _thrust_ctrl = generate_constant_ctrl(1.)

    def _ctrl_law(_t: float, _x: np.array, _u: np.array, _p_dict: dict, _k_dict: dict) -> np.array:
        return np.array(
            _aoa_ctrl(_t, _x, _u, _p_dict, _k_dict),
            _roll_ctrl(_t, _x, _u, _p_dict, _k_dict),
            _thrust_ctrl(_t, _x, _u, _p_dict, _k_dict),
        )
    return _ctrl_law


def eom(_t: float, _x: np.array, _u: np.array, _p_dict: dict, _k_dict: dict) -> np.array:
    _h = _x[0]
    _xn = _x[1]
    _xe = _x[2]
    _v = _x[3]
    _gam = _x[4]
    _psi = _x[5]
    _m = _x[6]

    _alpha = _u[0]
    _phi = _u[1]
    _f_thrust = _u[2]

    _g = _k_dict['mu'] / (_h + _k_dict['Re'])**2
    _mach = _v / atm.speed_of_sound(_h)
    _qdyn = 0.5 * atm.density(_h) * _v**2
    _cl_alpha = float(cl_alpha_table(_mach))
    _lift = _qdyn * _k_dict['s_ref'] * _cl_alpha * _alpha
    _drag = _qdyn * _k_dict['s_ref'] * (float(cd0_table(_mach)) + _k_dict['eta'] * _cl_alpha * _alpha ** 2)
    _thrust = _f_thrust * float(thrust_table((_mach, _h)))

    _dh = _v * np.sin(_gam)
    _dxn = _v * np.cos(_gam) * np.cos(_psi)
    _dxe = _v * np.cos(_gam) * np.sin(_psi)

    _dv = (_thrust * np.cos(_alpha) - _drag) / _m - _g * np.sin(_gam)
    _dgam = (_thrust * np.sin(_alpha) + _lift) * np.cos(_phi) / (_m * _v) - _g / _v * np.cos(_gam)
    _dpsi = (_thrust * np.sin(_alpha) + _lift) * np.sin(_phi) / (_m * _v * np.cos(_gam))

    _dm = -_thrust / (_k_dict['Isp'] * _k_dict['mu'] / _k_dict['Re']**2)

    return np.array((_dh, _dxn, _dxe, _dv, _dgam, _dpsi, _dm))


