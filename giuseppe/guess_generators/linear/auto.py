from __future__ import annotations
from typing import Union, Optional

import numpy as np
from numpy.typing import ArrayLike

from giuseppe.io.solution import Solution
from giuseppe.problems import CompBVP, CompOCP, CompDualOCP, AdiffBVP, AdiffOCP, AdiffDualOCP
from giuseppe.problems.dual.utils import sift_ocp_and_dual
from .simple import update_linear_value
from ..constant import initialize_guess_for_auto, update_constant_value
from ..projection import project_to_nullspace

SUPPORTED_PROBLEMS = Union[CompBVP, CompOCP, CompDualOCP, AdiffBVP, AdiffOCP, AdiffDualOCP]


def auto_linear_guess(comp_prob: SUPPORTED_PROBLEMS, t_span: Union[float, ArrayLike] = 0.1,
                      constants: Optional[ArrayLike] = None, default: Union[float, Solution] = 0.1,
                      use_dynamics: bool = False, abs_tol: float = 1e-3, rel_tol: float = 1e-3) -> Solution:
    """
    Automatically generate guess where variables (excluding the indenpendent) with initial and terminal values fitted
    to the boundary conditions and dynamics

    Parameters
    ----------
    comp_prob : CompBVP, CompOCP, or CompDualOCP 
        the problem that the guess is for
    t_span : float or ArrayLike, default=0.1
        values for the independent variable, t
        if float, t = np.array([0., t_span])
    constants : ArrayLike, optional
        constant  values
    default : float, BVPSol, OCPSol, or DualOCPSol, default=0.1
        if float, default value for set variables
        if solution type, guess to modify from which unmodified values will be taken
    use_dynamics : bool, default=True
        specifies whether the values are fitted with respect to a linear approximation of the dynamics or BC alone
    abs_tol : float, default=1e-3
        absolute tolerance
    rel_tol : float, default=1e-3
        relative tolerance

    Returns
    -------
    guess : Solution

    """
    prob, dual = sift_ocp_and_dual(comp_prob)
    guess = initialize_guess_for_auto(comp_prob, t_span=t_span, constants=constants, default=default)

    # Project states, (controls, ) and parameters to BCs
    num_x = prob.num_states
    num_p = prob.num_parameters

    if isinstance(prob, CompBVP):
        def map_values(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            _x0 = values[:num_x]
            _xf = values[num_x:2 * num_x]
            _p = values[2 * num_x:2 * num_x + num_p]
            return _x0, _xf, _p

        if use_dynamics:
            def bc_func(values):
                _x0, _xf, _p = map_values(values)
                psi_0 = prob.boundary_conditions.initial(guess.t[0], _x0, _p, guess.k)
                dyn_res = _xf - _x0 - (guess.t[-1] - guess.t[0]) \
                    * np.asarray(prob.dynamics((guess.t[-1] + guess.t[0]) / 2, (_xf + _x0) / 2, _p, guess.k))
                psi_f = prob.boundary_conditions.terminal(guess.t[-1], _xf, _p, guess.k)
                return np.concatenate((np.asarray(psi_0), dyn_res, np.asarray(psi_f)))

        else:
            def bc_func(values):
                _x0, _xf, _p = map_values(values)
                psi_0 = prob.boundary_conditions.initial(guess.t[0], _x0, _p, guess.k)
                psi_f = prob.boundary_conditions.terminal(guess.t[-1], _xf, _p, guess.k)
                return np.concatenate((np.asarray(psi_0), np.asarray(psi_f)))

        values_guess = np.concatenate(guess.x[:, 0], guess.x[:, -1], guess.p)
        out_x0, out_xf, out_p = map_values(project_to_nullspace(bc_func, values_guess, abs_tol=abs_tol, rel_tol=rel_tol))
        update_linear_value(guess, 'x', out_x0, out_xf)
        update_constant_value(guess, 'p', out_p)

    elif isinstance(prob, CompOCP):

        num_u = prob.num_controls

        def map_values(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            _x0 = values[:num_x]
            _xf = values[num_x:2 * num_x]
            _u0 = values[2 * num_x:2 * num_x + num_u]
            _uf = values[2 * num_x + num_u:2 * (num_x + num_u)]
            _p = values[2 * (num_x + num_u):2 * (num_x + num_u) + num_p]
            return _x0, _xf, _u0, _uf, _p

        if use_dynamics:
            def bc_func(values):
                _x0, _xf, _u0, _uf, _p = map_values(values)
                psi_0 = prob.boundary_conditions.initial(guess.t[0], _x0, _u0, _p, guess.k)
                dyn_res = _xf - _x0 - (guess.t[-1] - guess.t[0]) * np.asarray(
                    prob.dynamics((guess.t[-1] + guess.t[0]) / 2, (_xf + _x0) / 2, (_uf + _u0) / 2, _p, guess.k))
                psi_f = prob.boundary_conditions.terminal(guess.t[-1], _xf, _uf, _p, guess.k)
                return np.concatenate((np.asarray(psi_0), dyn_res, np.asarray(psi_f)))

        else:
            def bc_func(values):
                _x0, _xf, _u0, _uf, _p = map_values(values)
                psi_0 = prob.boundary_conditions.initial(guess.t[0], _x0, _u0, _p, guess.k)
                psi_f = prob.boundary_conditions.terminal(guess.t[-1], _xf, _uf, _p, guess.k)
                return np.concatenate((np.asarray(psi_0), np.asarray(psi_f)))

        values_guess = np.concatenate((guess.x[:, 0], guess.x[:, -1], guess.u[:, 0], guess.u[:, -1], guess.p))
        out_x0, out_xf, out_u0, out_uf, out_p = map_values(
                project_to_nullspace(bc_func, values_guess, abs_tol=abs_tol, rel_tol=rel_tol))
        update_linear_value(guess, 'x', out_x0, out_xf)
        update_linear_value(guess, 'u', out_u0, out_uf)
        update_constant_value(guess, 'p', out_p)

        if dual is not None:

            num_lam = dual.num_costates
            num_nu0 = dual.num_initial_adjoints
            num_nuf = dual.num_terminal_adjoints

            ocp_bc = prob.boundary_conditions
            dual_bc = dual.adjoined_boundary_conditions

            def map_dual_values(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                                                             np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                _x0 = values[:num_x]
                _xf = values[num_x:2 * num_x]
                _lam0 = values[2 * num_x:2 * num_x + num_lam]
                _lamf = values[2 * num_x + num_lam:2 * (num_x + num_lam)]
                _u0 = values[2 * (num_x + num_lam):2 * (num_x + num_lam) + num_u]
                _uf = values[2 * (num_x + num_lam) + num_u:2 * (num_x + num_lam + num_u)]
                _p = values[2 * (num_x + num_lam + num_u):num_x + num_lam + num_u + num_p]
                _nu0 = values[2 * (num_x + num_lam + num_u) + num_p:2 * (num_x + num_lam + num_u) + num_p + num_nu0]
                _nuf = values[2 * (num_x + num_lam + num_u) + num_p + num_nu0:
                              2 * (num_x + num_lam + num_u) + num_p + num_nu0 + num_nuf]
                return _x0, _xf, _lam0, _lamf, _u0, _uf, _p, _nu0, _nuf

            if use_dynamics:
                def dual_bc_func(values):
                    _x0, _xf, _lam0, _lamf, _u0, _uf, _p, _nu0, _nuf = map_dual_values(values)

                    psi_0 = ocp_bc.initial(guess.t[0], _x0, _u0, _p, guess.k)
                    dyn_res = _xf - _x0 - (guess.t[-1] - guess.t[0]) * np.asarray(prob.dynamics(
                        (guess.t[-1] + guess.t[0]) / 2, (_xf + _x0) / 2, (_uf + _u0) / 2, _p, guess.k))
                    psi_f = ocp_bc.terminal(guess.t[-1], _xf, _uf, _p, guess.k)

                    adj_bc0 = dual_bc.initial(guess.t[0], _x0, _lam0, _u0, _p, _nu0, guess.k)
                    costate_dyn_res = _lamf - _lam0 - (guess.t[-1] - guess.t[0]) \
                        * np.asarray(dual.costate_dynamics((guess.t[-1] + guess.t[0]) / 2, (_xf + _x0) / 2,
                                                           (_lamf + _lam0) / 2, (_uf + _u0) / 2, _p, guess.k))
                    adj_bcf = dual_bc.terminal(guess.t[-1], _xf, _lamf, _uf, _p, _nuf, guess.k)

                    return np.concatenate((
                        np.asarray(psi_0), dyn_res, np.asarray(psi_f),
                        np.asarray(adj_bc0), costate_dyn_res, np.asarray(adj_bcf)
                    ))

            else:
                def dual_bc_func(values):
                    _x0, _xf, _lam0, _lamf, _u0, _uf, _p, _nu0, _nuf = map_dual_values(values)

                    psi_0 = ocp_bc.initial(guess.t[0], _x0, _u0, _p, guess.k)
                    psi_f = ocp_bc.terminal(guess.t[-1], _xf, _uf, _p, guess.k)

                    adj_bc0 = dual_bc.initial(guess.t[0], _x0, _lam0, _u0, _p, _nu0, guess.k)
                    adj_bcf = dual_bc.terminal(guess.t[-1], _xf, _lamf, _uf, _p, _nuf, guess.k)

                    return np.concatenate((
                        np.asarray(psi_0), np.asarray(psi_f),
                        np.asarray(adj_bc0), np.asarray(adj_bcf)
                    ))

            values_guess = np.concatenate(
                    (guess.x[:, 0], guess.lam[:, 0], guess.u[:, 0], guess.x[:, -1], guess.lam[:, -1], guess.u[:, -1],
                     guess.p, guess.nu0, guess.nuf)
            )
            out_x0, out_xf, out_lam0, out_lamf, out_u0, out_uf, out_p, out_nu0, out_nuf = map_dual_values(
                    project_to_nullspace(dual_bc_func, values_guess, abs_tol=abs_tol, rel_tol=rel_tol)
            )
            update_linear_value(guess, 'x', out_x0, out_xf)
            update_linear_value(guess, 'lam', out_lam0, out_lamf)
            update_linear_value(guess, 'u', out_u0, out_uf)
            update_constant_value(guess, 'p', out_p)
            update_constant_value(guess, 'nu0', out_nu0)
            update_constant_value(guess, 'nuf', out_nuf)

    elif isinstance(prob, AdiffBVP):
        def map_values(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            _x0 = values[:num_x]
            _xf = values[num_x:2 * num_x]
            _p = values[2 * num_x:2 * num_x + num_p]
            return _x0, _xf, _p

        if use_dynamics:
            def bc_func(values):
                _x0, _xf, _p = map_values(values)
                psi_0 = prob.ca_boundary_conditions.initial(guess.t[0], _x0, _p, guess.k)
                dyn_res = _xf - _x0 - (guess.t[-1] - guess.t[0]) \
                    * np.asarray(prob.ca_dynamics((guess.t[-1] + guess.t[0]) / 2, (_xf + _x0) / 2, _p, guess.k)).flatten()
                psi_f = prob.ca_boundary_conditions.terminal(guess.t[-1], _xf, _p, guess.k)
                return np.concatenate((np.asarray(psi_0).flatten(), dyn_res, np.asarray(psi_f).flatten()))

        else:
            def bc_func(values):
                _x0, _xf, _p = map_values(values)
                psi_0 = prob.ca_boundary_conditions.initial(guess.t[0], _x0, _p, guess.k)
                psi_f = prob.ca_boundary_conditions.terminal(guess.t[-1], _xf, _p, guess.k)
                return np.concatenate((np.asarray(psi_0).flatten(), np.asarray(psi_f).flatten()))

        values_guess = np.concatenate(guess.x[:, 0], guess.x[:, -1], guess.p)
        out_x0, out_xf, out_p = map_values(project_to_nullspace(bc_func, values_guess, abs_tol=abs_tol, rel_tol=rel_tol))
        update_linear_value(guess, 'x', out_x0, out_xf)
        update_constant_value(guess, 'p', out_p)

    elif isinstance(prob, AdiffOCP):

        num_u = prob.num_controls

        def map_values(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            _x0 = values[:num_x]
            _xf = values[num_x:2 * num_x]
            _u0 = values[2 * num_x:2 * num_x + num_u]
            _uf = values[2 * num_x + num_u:2 * (num_x + num_u)]
            _p = values[2 * (num_x + num_u):2 * (num_x + num_u) + num_p]
            return _x0, _xf, _u0, _uf, _p

        if use_dynamics:
            def bc_func(values):
                _x0, _xf, _u0, _uf, _p = map_values(values)
                psi_0 = prob.ca_boundary_conditions.initial(guess.t[0], _x0, _u0, _p, guess.k)
                dyn_res = _xf - _x0 - (guess.t[-1] - guess.t[0]) * np.asarray(
                    prob.ca_dynamics((guess.t[-1] + guess.t[0]) / 2, (_xf + _x0) / 2, (_uf + _u0) / 2, _p, guess.k)).flatten()
                psi_f = prob.ca_boundary_conditions.terminal(guess.t[-1], _xf, _uf, _p, guess.k)
                return np.concatenate((np.asarray(psi_0).flatten(), dyn_res, np.asarray(psi_f).flatten()))

        else:
            def bc_func(values):
                _x0, _xf, _u0, _uf, _p = map_values(values)
                psi_0 = prob.ca_boundary_conditions.initial(guess.t[0], _x0, _u0, _p, guess.k)
                psi_f = prob.ca_boundary_conditions.terminal(guess.t[-1], _xf, _uf, _p, guess.k)
                return np.concatenate((np.asarray(psi_0).flatten(), np.asarray(psi_f).flatten()))

        values_guess = np.concatenate((guess.x[:, 0], guess.x[:, -1], guess.u[:, 0], guess.u[:, -1], guess.p))
        out_x0, out_xf, out_u0, out_uf, out_p = map_values(
                project_to_nullspace(bc_func, values_guess, abs_tol=abs_tol, rel_tol=rel_tol))
        update_linear_value(guess, 'x', out_x0, out_xf)
        update_linear_value(guess, 'u', out_u0, out_uf)
        update_constant_value(guess, 'p', out_p)

        if dual is not None:

            num_lam = dual.num_costates
            num_nu0 = dual.num_initial_adjoints
            num_nuf = dual.num_terminal_adjoints

            ocp_bc = prob.ca_boundary_conditions
            dual_bc = dual.ca_adj_boundary_conditions

            def map_dual_values(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                                                             np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                _x0 = values[:num_x]
                _xf = values[num_x:2 * num_x]
                _lam0 = values[2 * num_x:2 * num_x + num_lam]
                _lamf = values[2 * num_x + num_lam:2 * (num_x + num_lam)]
                _u0 = values[2 * (num_x + num_lam):2 * (num_x + num_lam) + num_u]
                _uf = values[2 * (num_x + num_lam) + num_u:2 * (num_x + num_lam + num_u)]
                _p = values[2 * (num_x + num_lam + num_u):num_x + num_lam + num_u + num_p]
                _nu0 = values[2 * (num_x + num_lam + num_u) + num_p:2 * (num_x + num_lam + num_u) + num_p + num_nu0]
                _nuf = values[2 * (num_x + num_lam + num_u) + num_p + num_nu0:
                              2 * (num_x + num_lam + num_u) + num_p + num_nu0 + num_nuf]
                return _x0, _xf, _lam0, _lamf, _u0, _uf, _p, _nu0, _nuf

            if use_dynamics:
                def dual_bc_func(values):
                    _x0, _xf, _lam0, _lamf, _u0, _uf, _p, _nu0, _nuf = map_dual_values(values)

                    psi_0 = ocp_bc.initial(guess.t[0], _x0, _u0, _p, guess.k)
                    dyn_res = _xf - _x0 - (guess.t[-1] - guess.t[0]) * np.asarray(prob.dynamics(
                        (guess.t[-1] + guess.t[0]) / 2, (_xf + _x0) / 2, (_uf + _u0) / 2, _p, guess.k)).flatten()
                    psi_f = ocp_bc.terminal(guess.t[-1], _xf, _uf, _p, guess.k)

                    adj_bc0 = dual_bc.initial(guess.t[0], _x0, _lam0, _u0, _p, _nu0, guess.k)
                    costate_dyn_res = _lamf - _lam0 - (guess.t[-1] - guess.t[0]) \
                        * np.asarray(dual.costate_dynamics((guess.t[-1] + guess.t[0]) / 2, (_xf + _x0) / 2,
                                                           (_lamf + _lam0) / 2, (_uf + _u0) / 2, _p, guess.k)).flatten()
                    adj_bcf = dual_bc.terminal(guess.t[-1], _xf, _lamf, _uf, _p, _nuf, guess.k)

                    return np.concatenate((
                        np.asarray(psi_0).flatten(), dyn_res, np.asarray(psi_f).flatten(),
                        np.asarray(adj_bc0).flatten(), costate_dyn_res, np.asarray(adj_bcf).flatten()
                    ))

            else:
                def dual_bc_func(values):
                    _x0, _xf, _lam0, _lamf, _u0, _uf, _p, _nu0, _nuf = map_dual_values(values)

                    psi_0 = ocp_bc.initial(guess.t[0], _x0, _u0, _p, guess.k)
                    psi_f = ocp_bc.terminal(guess.t[-1], _xf, _uf, _p, guess.k)

                    adj_bc0 = dual_bc.initial(guess.t[0], _x0, _lam0, _u0, _p, _nu0, guess.k)
                    adj_bcf = dual_bc.terminal(guess.t[-1], _xf, _lamf, _uf, _p, _nuf, guess.k)

                    return np.concatenate((
                        np.asarray(psi_0).flatten(), np.asarray(psi_f).flatten(),
                        np.asarray(adj_bc0).flatten(), np.asarray(adj_bcf).flatten()
                    ))

            values_guess = np.concatenate(
                    (guess.x[:, 0], guess.lam[:, 0], guess.u[:, 0], guess.x[:, -1], guess.lam[:, -1], guess.u[:, -1],
                     guess.p, guess.nu0, guess.nuf)
            )
            out_x0, out_xf, out_lam0, out_lamf, out_u0, out_uf, out_p, out_nu0, out_nuf = map_dual_values(
                    project_to_nullspace(dual_bc_func, values_guess, abs_tol=abs_tol, rel_tol=rel_tol)
            )
            update_linear_value(guess, 'x', out_x0, out_xf)
            update_linear_value(guess, 'lam', out_lam0, out_lamf)
            update_linear_value(guess, 'u', out_u0, out_uf)
            update_constant_value(guess, 'p', out_p)
            update_constant_value(guess, 'nu0', out_nu0)
            update_constant_value(guess, 'nuf', out_nuf)

    else:
        raise ValueError(f'Problem type {type(comp_prob)} not supported')

    return guess
