import numpy as np
import scipy as sp

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
    (0.9116837499743236, 0.3095782922169954),
    (1.1803863771747833, 0.3031941334757513),
    (2.282516484193661, 0.6162726208327338),
    (2.282516484193661, 0.6162726208327338),
    (2.9731990633280607, 0.726471252798718),
    (4.633511698128711, 0.8674650288601767),
    (6.813399954809686, 1.018910091818499),
    (6.813399954809686, 1.018910091818499),
))


# Fit Aero Model with Nonlinear Least Squares Estimate -----------------------------------------------------------------
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

if __name__ == '__main__':
    from matplotlib import pyplot as plt

    mach_vals = np.linspace(0.4, 7.0, 100)
    cla_vals = exponential_model(mach_vals, cla_coefficients[0], cla_coefficients[1], cla_coefficients[2])
    cd0_vals = exponential_model(mach_vals, cd0_coefficients[0], cd0_coefficients[1], cd0_coefficients[2])
    cdl_vals = exponential_model(mach_vals, cdl_coefficients[0], cdl_coefficients[1], cdl_coefficients[2])

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
