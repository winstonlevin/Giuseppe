from typing import Union, Optional
from warnings import warn

import numpy as np
# from scipy.integrate import solve_bvp

from giuseppe.data_classes import Solution
from giuseppe.problems.protocols import BVP, Dual
from giuseppe.problems.conversions import convert_dual_to_bvp, vectorize
from .scipy_bvp_problem import SciPyBVP
from .scipy_types import _scipy_bvp_sol
from .modified_solve_bvp import solve_bvp


class SciPySolver:
    """
    Class to use SciPy's BVP solver from scipy.integrate.solve_bvp

    The class takes in a supported problem type and wraps the boundary condition and dynamics functions to work.
    The class will also generate pre- and post-processing methods so that it can take in and output solutions in
    Giuseppe's native formats.

    """

    def __init__(self, prob: Union[BVP, Dual], use_jit_compile: bool = True, perform_vectorize: bool = True,
                 tol: float = 0.001, bc_tol: float = 0.001, max_nodes: int = 1000, node_buffer: int = 20,
                 verbose: Union[bool, int] = False, embed_continuation: bool = False,
                 max_mesh_iter: int = 10, max_newton_iter: int = 8):
        """
        Initialize SciPySolver

        Parameters
        ----------
        prob : BVP, Dual
            the BVP (or dualized OCP) to solve
        tol : float, default=0.001
            sets `tol` kwarg for `scipy.integrate.solve_bvp`
        bc_tol : float, default=0.001
            sets `bc_tol` kwarg for `scipy.integrate.solve_bvp`
        max_nodes: int, default=1000
            sets `max_nodes` kwarg for `scipy.integrate.solve_bvp`
        verbose : bool, default=False
            sets `verbose` kwarg for `scipy.integrate.solve_bvp`
        embed_continuation : bool, default=False
            embed a continuation into the BVP residual in order to modify the constants
        max_mesh_iter : int, optional
            number of mesh iterations in BVP solver
        max_newton_iter : int, optional
            number of residual-solving iterations in BVP solver
        """

        self.tol: float = tol
        self.bc_tol: float = bc_tol
        self.max_nodes: int = max_nodes
        self.node_buffer: int = node_buffer
        self.verbose: Union[bool, int] = verbose
        self.embed_continuation: bool = embed_continuation
        self.max_mesh_iter: int = max_mesh_iter
        self.max_newton_iter: int = max_newton_iter

        if prob.prob_class == 'dual':
            prob = convert_dual_to_bvp(prob, perform_vectorize=perform_vectorize)

        self.prob: SciPyBVP = SciPyBVP(prob, use_jit_compile=use_jit_compile)

    def solve(self, guess: Solution, constants: Optional[np.ndarray] = None,
              tol: Optional[float] = None, bc_tol: Optional[float] = None,
              max_nodes: Optional[int] = None, node_buffer: Optional[int] = None,
              verbose: Optional[Union[bool, int]] = None, embed_continuation: Optional[bool] = None,
              max_mesh_iter: Optional[int] = None, max_newton_iter: Optional[int] = None
        ) -> Solution:
        """
        Solve BVP (or dualized OCP) with instance of ScipySolveBVP

        Parameters
        ----------
        guess : Solution
            previous solution (or approximate solution) to serve as guess for BVP solver
        constants : np.ndarray, optional
            array of constants which define the problem numerically, if not given solver will use constants from guess
        tol : float, optional
            override solver default tolerance
        bc_tol : float, optional
            override solver default boundary value tolerance
        max_nodes : int, optional
            overide solver default maximum nodes
        node_buffer : int, optional
            override solver default node buffer
        verbose : bool or int, optional
            override solver default verbosity
        embed_continuation : bool, optional
            embed a continuation into the BVP residual in order to modify the constants
        max_mesh_iter : int, optional
            number of mesh iterations in BVP solver
        max_newton_iter : int, optional
            number of residual-solving iterations in BVP solver
        Returns
        -------
        solution : Solution
            solution to the BVP for given constants

        """

        tau_guess, x_guess, p_guess = self.prob.preprocess(guess)

        if constants is None:
            constants = guess.k

        constants = np.asarray(constants)

        if max_nodes is None:
            max_nodes = self.max_nodes
        if node_buffer is None:
            node_buffer = self.node_buffer
        if tol is None:
            tol = self.tol
        if bc_tol is None:
            bc_tol = self.bc_tol
        if verbose is None:
            verbose = self.verbose
        if embed_continuation is None:
            embed_continuation = self.embed_continuation
        if max_mesh_iter is None:
            max_mesh_iter = self.max_mesh_iter
        if max_newton_iter is None:
            max_newton_iter = self.max_newton_iter

        max_nodes = max(max_nodes, len(tau_guess) + node_buffer)

        if embed_continuation:
            # Add the continuation parameter as the final free constant with a constraint to be 1.
            # 0 corresponds to the constant values in the guess (assumed convergent), 1 corresponds to the constant
            # values passed separately into this function.
            constants0 = guess.k

            def _dynamics(_tau, _x, _p):
                # Remove continuation parameter (as expected in the compute_dynamics function)
                _constants = constants0 + _p[-1] * (constants - constants0)
                return self.prob.compute_dynamics(_tau, _x, _p[:-1], _constants)

            def _boundary_conditions(_x0, _xf, _p):
                # Add the equality constraint for the continuation parameter to be 1
                _constants = constants0 + _p[-1] * (constants - constants0)
                return np.append(self.prob.compute_boundary_conditions(_x0, _xf, _p[:-1], _constants), _p[-1] - 1.)

            _p_guess = np.append(p_guess, 0.)
        else:
            # Enforce the value of constants passed separately into this function
            def _dynamics(_tau, _x, _p):
                return self.prob.compute_dynamics(_tau, _x, _p, constants)

            def _boundary_conditions(_x0, _xf, _p):
                return self.prob.compute_boundary_conditions(_x0, _xf, _p, constants)

            _p_guess = p_guess

        try:
            sol: _scipy_bvp_sol = solve_bvp(
                    _dynamics, _boundary_conditions,
                    tau_guess, x_guess, _p_guess,
                    tol=tol, bc_tol=bc_tol, max_nodes=max_nodes, verbose=verbose,
                    max_mesh_iter=max_mesh_iter, max_newton_iter=max_newton_iter,
            )

            if embed_continuation:
                # Remove free parameter from solution, as expected in the post processor
                sol.p = sol.p[:-1]

        except RuntimeError as error:
            warn(error.args[0])
            sol: _scipy_bvp_sol = self._form_solution_when_solver_throws_exception(tau_guess, x_guess, p_guess)

        return self.prob.post_process(sol, constants)

    @staticmethod
    def _form_solution_when_solver_throws_exception(_x, _y, _p) -> _scipy_bvp_sol:
        class _FailedSciPySol:
            x: np.ndarray
            y: np.ndarray
            p: np.ndarray
            success: bool = False

        _failed = _FailedSciPySol
        _failed.x, _failed.y, _failed.p = _x, _y, _p

        return _failed
