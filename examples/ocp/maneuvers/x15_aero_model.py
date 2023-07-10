import numpy as np
import scipy as sp
import casadi as ca

from giuseppe.utils.examples import create_buffered_conditional_function, create_buffered_linear_interpolator

# Data extracted via WebPlot Digitizer from Fig. 8, 10, and 15.
# https://ntrs.nasa.gov/citations/19660010056

# CD(L = 0) vs. Mach
mach_cd0 = np.array((
    (0.5399081448475933, 0.06474513471493332),
    (0.858166590219988, 0.06667039964735248),
    (1.1048888147587757, 0.12283190268834179),
    (1.2583262347131463, 0.11082282439701441),
    (1.531639989754159, 0.0944628379787101),
    (1.9082818559268018, 0.07974313915876885),
    (2.4228494159275167, 0.06865189756542149),
    (3.0150292185163785, 0.059871926992428826),
    (4.00218617645482, 0.047012682206191606),
    (5.015773829025513, 0.04013510213077745),
    (5.995407242379656, 0.03725673574666566),
))

# d(CL)/dalpha [1/deg] vs. Mach
mach_cla = np.array((
    (0.6516548196150334, 0.06391848558678484),
    (0.8954026741684036, 0.07469319066636522),
    (1.1052168889558684, 0.07133984711861283),
    (1.3989567896583193, 0.05864516615175944),
    (1.7013119667780208, 0.05015916074314082),
    (2.2527482648046284, 0.039815135759676476),
    (3.0929428411155486, 0.03121359020085565),
    (4.027436599435695, 0.026501561466522633),
    (5.055760681184538, 0.023273135240583026),
    (5.803765939098619, 0.021608687279699237),
))

# d(CD)/d(CL**2) vs. Mach
# For CD = CD0 + eta * CLa * alpha**2 = CD0 + (eta/CLa) * CL**2
# This is equivalent to eta/CLa vs. Mach
mach_cdl = np.array((
    (0.5910738861614937, 0.3125937185465153),
    (0.9116837499743236, 0.3095782922169954),
    (1.1803863771747833, 0.3031941334757513),
    (2.282516484193661, 0.6162726208327338),
    (2.9731990633280607, 0.726471252798718),
    (4.633511698128711, 0.8674650288601767),
    (6.813399954809686, 1.018910091818499),
))

mach_MX = ca.MX.sym('M')


# Fit Supersonic Aero Model with Nonlinear Least Squares Estimate ------------------------------------------------------
def exponential_model(_mach, _c0, _c1, _c2):
    return _c0 + _c1 * np.exp(-_c2 * _mach)


# Fit CLa
cla_super_idces = np.where(mach_cla[:, 0] > 1.)
cla_result = sp.optimize.curve_fit(
    exponential_model, xdata=mach_cla[cla_super_idces[0], 0], ydata=mach_cla[cla_super_idces[0], 1]
)
cla_coefficients = cla_result[0]

# Fit CD0
cd0_super_idces = np.where(mach_cd0[:, 0] > 1.)
cd0_result = sp.optimize.curve_fit(
    exponential_model, xdata=mach_cd0[cd0_super_idces[0], 0], ydata=mach_cd0[cd0_super_idces[0], 1]
)
cd0_coefficients = cd0_result[0]

# Fit CD/CL**2 (CD/CL**2 = eta / CLa)
cdl_super_idces = np.where(mach_cdl[:, 0] > 1.)
cdl_result = sp.optimize.curve_fit(
    exponential_model, xdata=mach_cdl[cdl_super_idces[0], 0], ydata=mach_cdl[cdl_super_idces[0], 1]
)
cdl_coefficients = cdl_result[0]

cd0_supersonic_expr = exponential_model(mach_MX, cd0_coefficients[0], cd0_coefficients[1], cd0_coefficients[2])
cdl_supersonic_expr = exponential_model(mach_MX, cdl_coefficients[0], cdl_coefficients[1], cdl_coefficients[2])
cla_supersonic_expr = exponential_model(mach_MX, cla_coefficients[0], cla_coefficients[1], cla_coefficients[2])

cd0_supersonic_fun = ca.Function('CD0', (mach_MX,), (cd0_supersonic_expr,), ('M',), ('CD0',))
cdl_supersonic_fun = ca.Function('CDL', (mach_MX,), (cdl_supersonic_expr,), ('M',), ('CDL',))
cla_supersonic_fun = ca.Function('CLa', (mach_MX,), (cla_supersonic_expr,), ('M',), ('CLa',))

# # Generate Transonic Aero Model From Interpolant (PCHIP interpolation for consistent breakpoints) ----------------------
# cd0_data_pchip = sp.interpolate.PchipInterpolator(x=mach_cd0[:, 0], y=mach_cd0[:, 1])
# cdl_data_pchip = sp.interpolate.PchipInterpolator(x=mach_cdl[:, 0], y=mach_cdl[:, 1])
# cla_data_pchip = sp.interpolate.PchipInterpolator(x=mach_cla[:, 0], y=mach_cla[:, 1])
#
# boundary_thickness = 0.001
# mach_transonic = np.array((0.7, 0.8, 0.9, 1.0, 1.1))
# cd0_transonic = np.append(cd0_data_pchip(mach_transonic[:-2]),
#                           np.asarray(cd0_supersonic_fun(mach_transonic[-2:])).flatten())
# cdl_transonic = np.append(cdl_data_pchip(mach_transonic[:-2]),
#                           np.asarray(cdl_supersonic_fun(mach_transonic[-2:])).flatten())
# cla_transonic = np.append(cla_data_pchip(mach_transonic[:-2]),
#                           np.asarray(cla_supersonic_fun(mach_transonic[-2:])).flatten())
#
# # cd0_transonic_expr = create_buffered_linear_interpolator(
# #     x=mach_transonic, y=cd0_transonic, independent_var=mach_MX, boundary_thickness=0.15/2
# # )
#
# cd0_transonic_table = ca.interpolant('CD0', 'bspline', (mach_transonic,), cd0_transonic)
# cd0_transonic_expr = cd0_transonic_table(mach_MX)

# Generate Subsonic Aero Model PCHIP interpolation for consistent breakpoints) -----------------------------------------
cd0_data_pchip = sp.interpolate.PchipInterpolator(x=mach_cd0[:, 0], y=mach_cd0[:, 1])
cdl_data_pchip = sp.interpolate.PchipInterpolator(x=mach_cdl[:, 0], y=mach_cdl[:, 1])
cla_data_pchip = sp.interpolate.PchipInterpolator(x=mach_cla[:, 0], y=mach_cla[:, 1])

mach_subsonic = 0.65
mach_supersonic = 1.0
mach_transition = 0.925
boundary_thickness = mach_transition - mach_subsonic
cd0_subsonic = cd0_data_pchip(mach_subsonic)
cdl_subsonic = cdl_data_pchip(mach_subsonic)
cla_subsonic = cla_data_pchip(mach_subsonic)

breakpoints = np.array((0.0, mach_transition,))

cd0_expr = create_buffered_conditional_function(
    expr_list=[ca.MX(cd0_subsonic), cd0_supersonic_expr],
    break_points=breakpoints,
    independent_var=mach_MX,
    boundary_thickness=boundary_thickness,
)

cdl_expr = create_buffered_conditional_function(
    expr_list=[ca.MX(cdl_subsonic), cdl_supersonic_expr],
    break_points=breakpoints,
    independent_var=mach_MX,
    boundary_thickness=boundary_thickness,
)

cla_expr = create_buffered_conditional_function(
    expr_list=[ca.MX(cla_subsonic), cla_supersonic_expr],
    break_points=breakpoints,
    independent_var=mach_MX,
    boundary_thickness=boundary_thickness,
)

cd0_fun = ca.Function('cd0', (mach_MX,), (cd0_expr,), ('M',), ('cd0',))
cdl_fun = ca.Function('cdl', (mach_MX,), (cdl_expr,), ('M',), ('cdl',))
cla_fun = ca.Function('cla', (mach_MX,), (cla_expr,), ('M',), ('cla',))


if __name__ == '__main__':
    from matplotlib import pyplot as plt

    mach_vals = np.linspace(0.4, 7.0, 1000)
    # cla_vals = exponential_model(mach_vals, cla_coefficients[0], cla_coefficients[1], cla_coefficients[2])
    # cd0_vals = exponential_model(mach_vals, cd0_coefficients[0], cd0_coefficients[1], cd0_coefficients[2])
    # cdl_vals = exponential_model(mach_vals, cdl_coefficients[0], cdl_coefficients[1], cdl_coefficients[2])

    cd0_vals = np.asarray(cd0_fun(mach_vals)).flatten()
    cdl_vals = np.asarray(cdl_fun(mach_vals)).flatten()
    cla_vals = np.asarray(cla_fun(mach_vals)).flatten()

    fig_fit = plt.figure()

    ax_cla = fig_fit.add_subplot(311)
    ax_cla.grid()
    ax_cla.plot(mach_vals, cla_vals)
    ax_cla.plot(mach_cla[:, 0], mach_cla[:, 1], '*')
    ax_cla.set_xlabel('Mach')
    ax_cla.set_ylabel(r'$C_{L, \alpha}$')

    ax_cd0 = fig_fit.add_subplot(312)
    ax_cd0.grid()
    ax_cd0.plot(mach_vals, cd0_vals)
    ax_cd0.plot(mach_cd0[:, 0], mach_cd0[:, 1], '*')
    ax_cd0.set_xlabel('Mach')
    ax_cd0.set_ylabel(r'$C_{D,0}$')

    ax_eta = fig_fit.add_subplot(313)
    ax_eta.grid()
    ax_eta.plot(mach_vals, cdl_vals)
    ax_eta.plot(mach_cdl[:, 0], mach_cdl[:, 1], '*')
    ax_eta.set_xlabel('Mach')
    ax_eta.set_ylabel(r'$C_{D,L} = \eta / C_{L, \alpha}$')

    fig_fit.tight_layout()

    plt.show()
