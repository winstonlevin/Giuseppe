from typing import Union, Optional, Tuple
import casadi as ca
import numpy as np

CA_SYM = Union[ca.SX.sym, ca.MX.sym]


def fit_boundary_layer(f0: CA_SYM, f1: CA_SYM, x: CA_SYM, x0: float, x1: float,
                       free_variables: Optional[Tuple[Union[ca.SX, ca.MX]]] = None):
    # Fit coefficients for a fifth-order polynomial of the form:
    # p = C0 * [(x - x0)/dh]^0 + ... + C5 * [(x - x0)/dh]^5
    # Return C = [C0, C1, ..., C5]
    dx = x1 - x0
    dx2 = dx ** 2

    design_matrix = np.array((
        (1, 0, 0, 0, 0, 0),
        (0, 1 / dx, 0, 0, 0, 0),
        (0, 0, 2 / dx2, 0, 0, 0),
        (1, 1, 1, 1, 1, 1),
        (0, 1 / dx, 2 / dx, 3 / dx, 4 / dx, 5 / dx),
        (0, 0, 2 / dx2, 6 / dx2, 12 / dx2, 20 / dx2)
    ))

    f0x = ca.jacobian(f0, x)
    f0xx = ca.jacobian(f0x, x)

    f1x = ca.jacobian(f1, x)
    f1xx = ca.jacobian(f1x, x)

    if free_variables is not None:
        f0_fun = ca.Function('f0', (x, *free_variables), (f0,))
        f0x_fun = ca.Function('f0x', (x, *free_variables), (f0x,))
        f0xx_fun = ca.Function('f0xx', (x, *free_variables), (f0xx,))
        f1_fun = ca.Function('f1', (x, *free_variables), (f1,))
        f1x_fun = ca.Function('f1x', (x, *free_variables), (f1x,))
        f1xx_fun = ca.Function('f1xx', (x, *free_variables), (f1xx,))

        output = ca.vcat((
            f0_fun(x0, *free_variables), f0x_fun(x0, *free_variables), f0xx_fun(x0, *free_variables),
            f1_fun(x1, *free_variables), f1x_fun(x1, *free_variables), f1xx_fun(x1, *free_variables)
        ))

        coefficients = np.linalg.inv(design_matrix) @ output
    else:
        f0_fun = ca.Function('f0', (x,), (f0,))
        f0x_fun = ca.Function('f0x', (x,), (f0x,))
        f0xx_fun = ca.Function('f0xx', (x,), (f0xx,))
        f1_fun = ca.Function('f1', (x,), (f1,))
        f1x_fun = ca.Function('f1x', (x,), (f1x,))
        f1xx_fun = ca.Function('f1xx', (x,), (f1xx,))

        output = np.asarray((
            f0_fun(x0), f0x_fun(x0), f0xx_fun(x0),
            f1_fun(x1), f1x_fun(x1), f1xx_fun(x1)
        )).reshape((-1, 1))

        coefficients = np.linalg.solve(design_matrix, output)

    return coefficients, output


def create_conditional_function(expr_list: list[CA_SYM],
                                break_points: np.array,
                                independent_var: CA_SYM):
    expr_type = type(independent_var)
    ca_zero = expr_type(0.)
    expr_out = ca_zero
    idx_last = len(expr_list) - 1

    for idx, expr in enumerate(expr_list):
        # Switches for conditional function:
        # expression evaluated between switch0 and switch1
        # boundary (enforcing 2nd order continuity) evaluated between switch1 and switch2
        if idx == 0:
            expr_next = expr_list[idx + 1]
            switch0 = -ca.inf
            switch1 = break_points[idx + 1]
        elif idx == idx_last:
            expr_next = expr_list[idx_last]
            switch0 = break_points[idx]
            switch1 = +ca.inf
        else:
            expr_next = expr_list[idx + 1]
            switch0 = break_points[idx]
            switch1 = break_points[idx + 1]

        # Add expression and boundary expression to conditional function
        expr_out = ca.if_else(
            ca.logic_and(independent_var >= switch0, independent_var <= switch1),
            expr, expr_out
        )

    return expr_out


def create_buffered_conditional_function(expr_list: list[CA_SYM],
                                         break_points: np.array,
                                         independent_var: CA_SYM,
                                         boundary_thickness: float,
                                         free_variables: Optional[Tuple[Union[ca.SX, ca.MX]]] = None):
    expr_type = type(independent_var)
    ca_zero = expr_type(0.)
    expr_out = ca_zero
    idx_last = len(expr_list) - 1

    for idx, expr in enumerate(expr_list):
        # Switches for conditional function:
        # expression evaluated between switch0 and switch1
        # boundary (enforcing 2nd order continuity) evaluated between switch1 and switch2
        if idx == 0:
            expr_next = expr_list[idx + 1]
            switch0 = -ca.inf
            switch1 = break_points[idx + 1] - 0.5 * boundary_thickness
            switch2 = break_points[idx + 1] + 0.5 * boundary_thickness
        elif idx == idx_last:
            expr_next = expr_list[idx_last]
            switch0 = break_points[idx] + 0.5 * boundary_thickness
            switch1 = +ca.inf
            switch2 = +ca.inf
        else:
            expr_next = expr_list[idx + 1]
            switch0 = break_points[idx] + 0.5 * boundary_thickness
            switch1 = break_points[idx + 1] - 0.5 * boundary_thickness
            switch2 = break_points[idx + 1] + 0.5 * boundary_thickness

        # Create boundary function tying 1st and 2nd derivatives between two expressions via 5th order polynomial
        coeffs, _ = fit_boundary_layer(expr, expr_next, independent_var, switch1, switch2, free_variables)
        x_normalized = (independent_var - switch1) / boundary_thickness
        boundary_expr = ca_zero
        if free_variables is not None:
            for power, coeff in enumerate(ca.vertsplit(coeffs)):
                boundary_expr += coeff * x_normalized ** power
        else:
            for power, coeff in enumerate(coeffs):
                boundary_expr += coeff * x_normalized ** power

        # Add expression and boundary expression to conditional function
        expr_out += ca.if_else(
            ca.logic_and(independent_var >= switch0, independent_var <= switch1),
            expr, ca_zero
        )
        expr_out += ca.if_else(
            ca.logic_and(independent_var > switch1, independent_var < switch2),
            boundary_expr, ca_zero
        )

    return expr_out


def create_buffered_linear_interpolator(x: np.array, y: Union[list[CA_SYM], np.array],
                                        independent_var: CA_SYM, boundary_thickness: float,
                                        free_variables: Optional[Tuple[Union[ca.SX, ca.MX]]] = None):
    expr_type = type(independent_var)
    expr_list = [expr_type.nan()] * (len(y) - 1)
    for idx, (x1, x2, y1, y2) in enumerate(zip(x[:-1], x[1:], y[:-1], y[1:])):
        expr_list[idx] = y1 + (y2 - y1) / (x2 - x1) * (independent_var - x1)
    expr_out = create_buffered_conditional_function(expr_list, x, independent_var, boundary_thickness, free_variables)
    return expr_out
#
#
# def create_quintic_hermitian_interpolator(x: np.array, y: Union[list[CA_SYM], np.array],
#                                           independent_var: CA_SYM):
#     expr_type = type(independent_var)
#     expr_list = [expr_type.nan()] * (len(y) - 1)
#     expr_out = create_conditional_function(expr_list, break_points, independent_var)


def create_buffered_2d_linear_interpolator(x: np.array, y: np.array, z: np.array,
                                           independent_vars: list[CA_SYM],
                                           boundary_thicknesses: np.array):
    '''

    Parameters
    ----------
    x : np.array, 1D array of dimension n, representing input data for x
    y : np.array, 1D array of dimension m representing input data for y
    z : np.array, 2D array of dimension n by m representing output data for grid of (x, y)
    independent_vars : list[CA_SYM], list of independent variables x, y
    boundary_thicknesses : np.array, boundary layer thickness for x, y respectively

    Returns
    -------
    expr_out : Union[ca.SX, ca.MX] 2-D interpolant expression of x, y.
    '''

    m_data = len(y)
    expr_type = type(independent_vars[0])

    # Generate m 1-D buffered linear interpolants of the first variable
    expr_list = [expr_type.nan()] * (m_data - 1)
    for idx in range(m_data - 1):
        expr_list[idx] = create_buffered_linear_interpolator(
            x, z[:, idx],
            independent_vars[0], boundary_thicknesses[0]
        )

    # Interpolate between the 1-D linear interpolants for the second variable
    expr_out = create_buffered_linear_interpolator(
        y, expr_list, independent_vars[1], boundary_thicknesses[1], free_variables=(independent_vars[0],)
    )

    return expr_out
