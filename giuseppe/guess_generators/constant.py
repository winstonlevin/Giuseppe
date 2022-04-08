from typing import Union

import numpy as np
from numpy.typing import ArrayLike

from ..problems.dual.utils import sift_ocp_and_dual
from ..problems.dual.solution import Solution
from ..problems.ocp import CompOCP
from ..problems.typing import AnyProblem, AnySolution


def generate_constant_guess(comp_prob: AnyProblem, t_span: Union[float, ArrayLike] = 0.1, constant: float = 1.) \
        -> AnySolution:

    prob, dual = sift_ocp_and_dual(comp_prob)

    data = {'converged': False}

    if isinstance(t_span, float) or isinstance(t_span, int):
        data['t'] = np.array([0., t_span], dtype=float)
    else:
        data['t'] = np.array(t_span, dtype=float)

    num_t_steps = len(data['t'])

    if prob is not None:
        data['x'] = np.ones((prob.num_states, num_t_steps)) * constant
        data['p'] = np.ones((prob.num_parameters,)) * constant
        data['k'] = prob.default_values

    if isinstance(prob, CompOCP):
        data['u'] = np.ones((prob.num_controls, num_t_steps)) * constant

    if dual is not None:
        data['lam'] = np.ones((dual.num_costates, num_t_steps)) * constant
        data['nu0'] = np.ones((dual.num_initial_adjoints,)) * constant
        data['nuf'] = np.ones((dual.num_terminal_adjoints,)) * constant

    return Solution(**data)
