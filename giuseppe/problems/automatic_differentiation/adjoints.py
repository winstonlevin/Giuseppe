from copy import deepcopy
from typing import Union

import casadi as ca
import numpy as np

from giuseppe.problems.protocols import OCP, VectorizedAdjoints
from .input import ADiffInputProb
from .ocp import ADiffOCP
from .utils import lambdify_ca, get_names
from ..symbolic.ocp import SymOCP, StrInputProb


class ADiffAdjoints(VectorizedAdjoints):
    def __init__(self, source_ocp: Union[ADiffOCP, ADiffInputProb, SymOCP, OCP, StrInputProb]):

        if isinstance(source_ocp, ADiffOCP):
            self.source_ocp = deepcopy(source_ocp)
        else:
            self.source_ocp = ADiffOCP(source_ocp)

        self.annotations = self.source_ocp.annotations

        self.dtype = self.source_ocp.dtype
        self.num_states = self.source_ocp.num_states
        self.num_controls = self.source_ocp.num_controls
        self.num_parameters = self.source_ocp.num_parameters
        self.num_constants = self.source_ocp.num_constants
        self.num_costates = self.num_states + self.num_parameters
        self.num_initial_adjoints = self.source_ocp.ca_initial_boundary_conditions.size_out(0)[0]
        self.num_terminal_adjoints = self.source_ocp.ca_terminal_boundary_conditions.size_out(0)[0]
        self.num_adjoints = self.num_initial_adjoints + self.num_terminal_adjoints

        self.independent = self.source_ocp.independent
        self.states = self.source_ocp.states
        self.controls = self.source_ocp.controls
        self.parameters = self.source_ocp.parameters
        self.constants = self.source_ocp.constants

        self.costates = self.dtype.sym('_lam', self.num_costates)
        self.initial_adjoints = self.dtype.sym('_nu_0', self.num_initial_adjoints)
        self.terminal_adjoints = self.dtype.sym('_nu_f', self.num_terminal_adjoints)
        self.states_and_parameters = ca.vcat((self.states, self.parameters))

        self.annotations.costates = get_names(self.costates)
        self.annotations.nu0 = get_names(self.initial_adjoints)
        self.annotations.nuf = get_names(self.terminal_adjoints)

        self.arg_names = {
            'ocp_dynamic' : self.source_ocp.dyn_arg_names,
            'ocp_bc'      : self.source_ocp.bc_arg_names,
            'adj_initial' : ('t', 'x', 'lam', 'u', 'p', '_nu_0', 'k'),
            'adj_dynamic' : ('t', 'x', 'lam', 'u', 'p', 'k'),
            'adj_terminal': ('t', 'x', 'lam', 'u', 'p', '_nu_f', 'k')
        }

        self.args = {
            'ocp_dynamic' : self.source_ocp.dyn_args,
            'ocp_bc'      : self.source_ocp.bc_args,
            'adj_initial' : (self.independent, self.states, self.costates, self.controls, self.parameters,
                             self.initial_adjoints, self.constants),
            'adj_dynamic' : (self.independent, self.states, self.costates, self.controls, self.parameters,
                             self.constants),
            'adj_terminal': (self.independent, self.states, self.costates, self.controls, self.parameters,
                             self.terminal_adjoints, self.constants)
        }

        psi_0 = self.source_ocp.ca_initial_boundary_conditions(*self.args['ocp_bc'])
        psi_f = self.source_ocp.ca_terminal_boundary_conditions(*self.args['ocp_bc'])
        phi_0 = self.source_ocp.ca_initial_cost(*self.args['ocp_bc'])
        phi_f = self.source_ocp.ca_terminal_cost(*self.args['ocp_bc'])
        lagrangian = self.source_ocp.ca_path_cost(*self.args['ocp_dynamic'])
        f = self.source_ocp.eom

        hamiltonian = lagrangian + ca.dot(self.costates, f)
        dh_dxp = ca.jacobian(hamiltonian, self.states_and_parameters)
        dh_du = ca.jacobian(hamiltonian, self.controls)
        d2h_du2 = ca.jacobian(dh_du, self.controls)
        dh_dt = ca.jacobian(hamiltonian, self.independent)
        phi_0_adj = phi_0 + ca.dot(self.initial_adjoints, psi_0)
        phi_f_adj = phi_f + ca.dot(self.terminal_adjoints, psi_f)

        self.ca_hamiltonian = ca.Function(
                'H', self.args['adj_dynamic'],
                (hamiltonian,),
                self.arg_names['adj_dynamic'], ('H',))
        self.ca_costate_dynamics = ca.Function(
                'lam_dot', self.args['adj_dynamic'],
                (-dh_dxp.T,),
                self.arg_names['adj_dynamic'], ('lam_dot',)
        )
        self.ca_dh_du = ca.Function(
                'dH_du', self.args['adj_dynamic'],
                (dh_du,),
                self.arg_names['adj_dynamic'], ('dH_du',))
        self.ca_d2h_du2 = ca.Function(
            'd2H_du2', self.args['adj_dynamic'],
            (d2h_du2,),
            self.arg_names['adj_dynamic'], ('d2H_du2',))
        self.ca_dh_dt = ca.Function(
                'dH_dt', self.args['adj_dynamic'],
                (dh_dt,),
                self.arg_names['adj_dynamic'], ('dH_dt',))

        self.ca_initial_adjoint_cost = ca.Function(
                'Phi_0_adj', self.args['adj_initial'],
                (phi_0_adj,),
                self.arg_names['adj_initial'], ('Phi_0_adj',))
        self.ca_terminal_adjoint_cost = ca.Function(
                'Phi_f_adj', self.args['adj_terminal'],
                (phi_f_adj,),
                self.arg_names['adj_terminal'], ('Phi_f_adj',))

        adj1 = ca.jacobian(phi_0_adj, self.independent) - hamiltonian
        adj2 = ca.jacobian(phi_0_adj, self.states_and_parameters).T + self.costates
        adj3 = ca.jacobian(phi_f_adj, self.independent) + hamiltonian
        adj4 = ca.jacobian(phi_f_adj, self.states_and_parameters).T - self.costates

        self.ca_initial_adjoint_boundary_conditions = ca.Function(
                'Psi_0_adj', self.args['adj_initial'],
                (ca.vertcat(adj1, adj2),),
                self.arg_names['adj_initial'], ('Psi_0_adj',)
        )
        self.ca_terminal_adjoint_boundary_conditions = ca.Function(
                'Psi_f_adj', self.args['adj_terminal'],
                (ca.vertcat(adj3, adj4),),
                self.arg_names['adj_terminal'], ('Psi_f_adj',)
        )

        def _compute_control_convexity(independent: float, states: np.ndarray, costates: np.ndarray,
                                       controls: np.ndarray, parameters: np.ndarray, constants: np.ndarray):
            _h_uu = np.asarray(self.ca_d2h_du2(independent, states, costates, controls, parameters, constants))
            if np.any(np.logical_or(np.isnan(_h_uu), np.isinf(_h_uu))):
                return np.nan
            else:
                return np.linalg.eigvals(_h_uu)

        self.compute_costate_dynamics = lambdify_ca(self.ca_costate_dynamics)

        self.compute_hamiltonian = lambdify_ca(self.ca_hamiltonian)
        self.compute_control_law = lambdify_ca(self.ca_dh_du)
        self.compute_control_convexity = _compute_control_convexity

        self.compute_initial_adjoint_boundary_conditions = lambdify_ca(self.ca_initial_adjoint_boundary_conditions)
        self.compute_terminal_adjoint_boundary_conditions = lambdify_ca(self.ca_terminal_adjoint_boundary_conditions)

        self.compute_costate_dynamics_vectorized, self.compute_hamiltonian_vectorized,\
            self.compute_control_law_vectorized, self.compute_control_convexity_vectorized = self.vectorize()

    def vectorize(self):
        _compute_costate_dynamics = self.ca_costate_dynamics

        def _compute_costate_dynamics_vectorized(
                independent: np.ndarray, states: np.ndarray, costates: np.ndarray, controls: np.ndarray,
                parameters: np.ndarray, constants: np.ndarray
        ) -> np.ndarray:

            map_size = len(independent)
            _costate_dynamics_mapped = _compute_costate_dynamics.map(map_size)

            return np.array(_costate_dynamics_mapped(independent, states, costates, controls, parameters, constants))

        _compute_hamiltonian = self.ca_hamiltonian

        def _compute_hamiltonian_vectorized(
                independent: np.ndarray, states: np.ndarray, costates: np.ndarray, controls: np.ndarray,
                parameters: np.ndarray, constants: np.ndarray
        ) -> np.ndarray:

            map_size = len(independent)
            _hamiltonian_mapped = _compute_hamiltonian.map(map_size)

            return np.array(_hamiltonian_mapped(independent, states, costates, controls, parameters, constants))

        _compute_control_law = self.ca_dh_du

        def _compute_control_law_vectorized(
                independent: np.ndarray, states: np.ndarray, costates: np.ndarray, controls: np.ndarray,
                parameters: np.ndarray, constants: np.ndarray
        ) -> np.ndarray:

            map_size = len(independent)
            _control_law_mapped = _compute_control_law.map(map_size)

            return np.array(_control_law_mapped(independent, states, costates, controls, parameters, constants))

        def _compute_control_convexity_vectorized(
                t: np.ndarray, x: np.ndarray, lam: np.ndarray, u: np.ndarray, p: np.ndarray, k: np.ndarray
        ) -> np.ndarray:
            # TODO vectorize eigenvalue calculation
            eig_h_uu = np.array([
                np.asarray(self.ca_d2h_du2(ti, xi, lam_i, ui, p, k)) for ti, xi, lam_i, ui in zip(t, x.T, lam.T, u.T)
            ]).T
            return eig_h_uu

        return _compute_costate_dynamics_vectorized, _compute_hamiltonian_vectorized,\
            _compute_control_law_vectorized, _compute_control_convexity_vectorized
