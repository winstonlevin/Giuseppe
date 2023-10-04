import numpy as np
import casadi as ca

from giuseppe.utils.examples import Atmosphere1976, create_buffered_conditional_function

raw_aero_data = [
    {  # Mach 10
        'M': 10.,
        'CL': np.array((
            (1.9537433086596838, -0.01497323463873168),
            (4.074481981405119, 0.0020720743795301644),
            (6.06818123607305, 0.027275055707809304),
            (7.937914606971804, 0.04834157211279877),
            (9.931613861639734, 0.0735445534410778),
        )),
        'CD': np.array((
            (1.9871531346351503, 0.09147225077081189),
            (3.9933196300102782, 0.0919655704008222),
            (5.998972250770814, 0.09491778006166496),
            (8.004110996916756, 0.10032887975334015),
            (9.88386433710175, 0.10570914696813971),
        ))
    },
    {  # Mach 6
        'M': 6.,
        'CL': np.array((
            (2.076684680992754, -0.006738723971006388),
            (3.9443690290192865, 0.022523883922854315),
            (5.938068283687217, 0.047726865251133344),
            (7.931767538355151, 0.07292984657941248),
            (9.923417770150861, 0.10632891939656275),
        )),
        'CD': np.array((
            (1.9861253854059626, 0.09639003083247677),
            (3.9928057553956844, 0.09442446043165464),
            (5.87358684480987, 0.09488694758478933),
            (8.004110996916756, 0.10032887975334015),
        ))
    },
    {  # Mach 4.5
        'M': 4.5,
        'CL': np.array((
            (2.076684680992754, -0.006738723971006388),
            (3.9443690290192865, 0.022523883922854315),
            (5.938068283687217, 0.047726865251133344),
            (7.931767538355151, 0.07292984657941248),
            (9.923417770150861, 0.10632891939656275),
        )),
        'CD': np.array((
            (2.56880733944954, 0.10119047619047628),
            (4.311926605504587, 0.09722222222222232),
            (5.8715596330275215, 0.09325396825396837),
            (7.706422018348624, 0.09722222222222232),
            (9.449541284403667, 0.09722222222222232),
        )),
    },
    {  # Mach 3.5
        'M': 3.5,
        'CL': np.array((
            (4.327581185379351, -0.029365079365079372),
            (7.7286296781709645, 0.042063492063492136),
        )),
        'CD': np.array((
            (3.2110091743119256, 0.10912698412698418),
            (4.220183486238531, 0.1071428571428572),
            (5.8715596330275215, 0.1071428571428572),
            (7.706422018348624, 0.1071428571428572),
            (9.449541284403667, 0.11111111111111116),
        )),
    },
    {  # Mach 2.5
        'M': 2.5,
        'CL': np.array((
            (2.488714140090286, -0.07301587301587287),
            (7.731542158147665, 0.07380952380952388),
            (9.570773263433814, 0.12142857142857144),
        )),
        'CD': np.array((
            (2.844036697247706, 0.12896825396825418),
            (4.220183486238531, 0.125),
            (6.146788990825689, 0.125),
            (7.706422018348624, 0.12698412698412698),
            (9.449541284403667, 0.13293650793650813),
        )),
    },
    {  # Mach 1.6
        'M': 1.6,
        'CL': np.array((
            (2.5775447793796413, -0.10476190476190483),
            (5.894859472841123, 0.053968253968254),
            (7.737367118101062, 0.13730158730158737),
            (9.486675404106595, 0.2047619047619048),
        )),
        'CD': np.array((
            (2.6605504587155977, 0.15277777777777768),
            (4.311926605504587, 0.14484126984127),
            (6.05504587155963, 0.14484126984127),
            (7.706422018348624, 0.14880952380952395),
            (9.449541284403667, 0.1607142857142858),
        )),
    },
    {  # Mach 1.2
        'M': 1.2,
        'CL': np.array((
            (4.09626610860227, -0.04334177772882475),
            (5.9048353342879185, 0.05292433087344439),
            (7.714506002863754, 0.15899328119837008),
            (9.296177993171057, 0.2358739949333628),
        )),
        'CD': np.array((
            (-1.8436080467229061, 0.15920398009950265),
            (0.24205061648280335, 0.1512437810945273),
            (2.0674886437378337, 0.14726368159204006),
            (4.0253082414016905, 0.15522388059701497),
            (5.983776768332254, 0.16716417910447778),
            (8.205061648280337, 0.19104477611940318),
        )),
    },
    {  # Mach 0.8
        'M': 0.8,
        'CL': np.array((
            (4.88159488930499, -0.05391562947461148),
            (6.575614054411282, 0.022854939971362453),
            (7.931490252230422, 0.09015310056173598),
        )),
        'CD': np.array((
            (-1.075924724205061, 0.06766169154228852),
            (1.0103828682673601, 0.06368159203980106),
            (2.8364698247891003, 0.06368159203980106),
            (4.924724205061649, 0.07164179104477619),
            (6.8831927319922155, 0.083582089552239),
            (8.973393900064893, 0.10348258706467672),
        ))
    }
]


# Fit functions
def fit_line(_x, _y):
    # Model: y = c0 + c1 * x
    _x_2d = _x.reshape((-1, 1))
    _design_matrix = np.hstack((np.ones(_x_2d.shape), _x_2d))
    _output = _y
    _coeffs = np.linalg.lstsq(_design_matrix, _output, rcond=None)[0]
    return _coeffs


def fit_parabola(_x, _y, _offset=True):
    # Model: y = c0 + c1 * x + c2 * x**2
    _x_2d = _x.reshape((-1, 1))
    _output = _y
    if _offset:
        _design_matrix = np.hstack((np.ones(_x_2d.shape), _x_2d, _x_2d**2))
    else:
        _design_matrix = np.hstack((np.ones(_x_2d.shape), np.zeros(_x_2d.shape), _x_2d ** 2))
    _coeffs = np.linalg.lstsq(_design_matrix, _output, rcond=None)[0]
    return _coeffs


# CL = CL0 + CLa * alpha @ each Mach number
# CD = CD0 + CD1 * CL + CD2 * CL**2 @ each Mach number
d2r = np.pi / 180.
n_mach = len(raw_aero_data)
machs = np.empty((n_mach,))
CL0_arr = np.empty((n_mach,))
CLa_arr = np.empty((n_mach,))
CD0_arr = np.empty((n_mach,))
CD1_arr = np.empty((n_mach,))
CD2_arr = np.empty((n_mach,))
for idx in range(n_mach):
    machs[idx] = raw_aero_data[idx]['M']

    # Fit CL Model
    alpha_vals = raw_aero_data[idx]['CL'][:, 0] * d2r
    CL_vals = raw_aero_data[idx]['CL'][:, 1]
    coeffs = fit_line(alpha_vals, CL_vals)
    CL0_arr[idx] = coeffs[0]
    CLa_arr[idx] = coeffs[1]

    # Fit CD Model
    alpha_vals = raw_aero_data[idx]['CD'][:, 0] * d2r
    CL_vals = CL0_arr[idx] + CLa_arr[idx] * alpha_vals
    CD_vals = raw_aero_data[idx]['CD'][:, 1]
    coeffs = fit_parabola(CL_vals, CD_vals)
    CD0_arr[idx] = coeffs[0]
    CD1_arr[idx] = coeffs[1]
    CD2_arr[idx] = coeffs[2]

if machs[-1] > machs[0]:
    lut_data = {
        'M': machs,
        'CL0': CL0_arr,
        'CLa': CLa_arr,
        'CD0': CD0_arr,
        'CD1': CD1_arr,
        'CD2': CD2_arr
    }
else:
    lut_data = {
        'M': np.flip(machs),
        'CL0': np.flip(CL0_arr),
        'CLa': np.flip(CLa_arr),
        'CD0': np.flip(CD0_arr),
        'CD1': np.flip(CD1_arr),
        'CD2': np.flip(CD2_arr)
    }

# Build aero look-up tables
mach_sym = ca.MX.sym('M')
interpolant_CL0 = ca.interpolant('CL0', 'bspline', (lut_data['M'],), lut_data['CL0'])
interpolant_CLa = ca.interpolant('CLa', 'bspline', (lut_data['M'],), lut_data['CLa'])
interpolant_CD0 = ca.interpolant('CD0', 'bspline', (lut_data['M'],), lut_data['CD0'])
interpolant_CD1 = ca.interpolant('CD1', 'bspline', (lut_data['M'],), lut_data['CD1'])
interpolant_CD2 = ca.interpolant('CD2', 'bspline', (lut_data['M'],), lut_data['CD2'])
interp_fun_CL0 = ca.Function('CL0', (mach_sym,), (interpolant_CL0(mach_sym),), ('M',), ('CL0',))
interp_fun_CLa = ca.Function('CLa', (mach_sym,), (interpolant_CLa(mach_sym),), ('M',), ('CLa',))
interp_fun_CD0 = ca.Function('CD0', (mach_sym,), (interpolant_CD0(mach_sym),), ('M',), ('CD0',))
interp_fun_CD1 = ca.Function('CD1', (mach_sym,), (interpolant_CD1(mach_sym),), ('M',), ('CD1',))
interp_fun_CD2 = ca.Function('CD2', (mach_sym,), (interpolant_CD2(mach_sym),), ('M',), ('CD2',))

# Since tables have a discontinuity at the endpoints of Mach number, create buffered conditional function with constant
# extrapolation.
mach_boundary_thickness = 0.05
CL0_expr = create_buffered_conditional_function(
    expr_list=[lut_data['CL0'][0], interp_fun_CL0(mach_sym), lut_data['CL0'][-1]],
    break_points=[-np.inf, lut_data['M'][0] + mach_boundary_thickness, lut_data['M'][-1] - mach_boundary_thickness],
    independent_var=mach_sym,
    boundary_thickness=0.1  # 0.1 Mach boundary thickness
)
CLa_expr = create_buffered_conditional_function(
    expr_list=[lut_data['CLa'][0], interp_fun_CLa(mach_sym), lut_data['CLa'][-1]],
    break_points=[-np.inf, lut_data['M'][0] + mach_boundary_thickness, lut_data['M'][-1] - mach_boundary_thickness],
    independent_var=mach_sym,
    boundary_thickness=0.1  # 0.1 Mach boundary thickness
)
CD0_expr = create_buffered_conditional_function(
    expr_list=[lut_data['CD0'][0], interp_fun_CD0(mach_sym), lut_data['CD0'][-1]],
    break_points=[-np.inf, lut_data['M'][0] + mach_boundary_thickness, lut_data['M'][-1] - mach_boundary_thickness],
    independent_var=mach_sym,
    boundary_thickness=0.1  # 0.1 Mach boundary thickness
)
CD1_expr = create_buffered_conditional_function(
    expr_list=[lut_data['CD1'][0], interp_fun_CD1(mach_sym), lut_data['CD1'][-1]],
    break_points=[-np.inf, lut_data['M'][0] + mach_boundary_thickness, lut_data['M'][-1] - mach_boundary_thickness],
    independent_var=mach_sym,
    boundary_thickness=0.1  # 0.1 Mach boundary thickness
)
CD2_expr = create_buffered_conditional_function(
    expr_list=[lut_data['CD2'][0], interp_fun_CD2(mach_sym), lut_data['CD2'][-1]],
    break_points=[-np.inf, lut_data['M'][0] + mach_boundary_thickness, lut_data['M'][-1] - mach_boundary_thickness],
    independent_var=mach_sym,
    boundary_thickness=0.1  # 0.1 Mach boundary thickness
)
CL0_fun = ca.Function('CL0', (mach_sym,), (CL0_expr,), ('M',), ('CL0',))
CLa_fun = ca.Function('CLa', (mach_sym,), (CLa_expr,), ('M',), ('CLa',))
CD0_fun = ca.Function('CD0', (mach_sym,), (CD0_expr,), ('M',), ('CD0',))
CD1_fun = ca.Function('CD1', (mach_sym,), (CD1_expr,), ('M',), ('CD1',))
CD2_fun = ca.Function('CD2', (mach_sym,), (CD2_expr,), ('M',), ('CD2',))

if __name__ == '__main__':
    from matplotlib import pyplot as plt
    import matplotlib as mpl

    cols = plt.rcParams['axes.prop_cycle'].by_key()['color']
    gradient = mpl.colormaps['viridis'].colors
    # Generate color gradient
    if len(machs) == 1:
        grad_idcs = np.array((0,), dtype=np.int32)
    else:
        grad_idcs = np.int32(np.floor(np.linspace(0, 255, len(machs))))


    def cols_gradient(n):
        return gradient[grad_idcs[n]]


    # Verify aero coefficient fit
    r2d = 180/np.pi
    alpha_vals = np.linspace(-2., 30., 200) * d2r

    fig_curves = plt.figure()
    ax_cl = fig_curves.add_subplot(311)
    ax_cl.grid()
    ax_cl.set_xlabel(r'$\alpha$ [deg]')
    ax_cl.set_ylabel(r'$C_L$')
    ax_cd = fig_curves.add_subplot(312)
    ax_cd.grid()
    ax_cd.set_xlabel(r'$\alpha$ [deg]')
    ax_cd.set_ylabel(r'$C_D$')
    ax_ld = fig_curves.add_subplot(313)
    ax_ld.grid()
    ax_ld.set_xlabel(r'$\alpha$ [deg]')
    ax_ld.set_ylabel(r'$L/D$')
    ax_ld.set_yticks((0, 1, 2, 3, 4))

    for idx in range(n_mach):
        lab = f'Mach {machs[idx]}'
        # if idx == 0 or idx == n_mach-1:
        #     lab = f'Mach {machs[idx]}'
        # else:
        #     lab = None

        CL_vals = CL0_arr[idx] + CLa_arr[idx] * alpha_vals
        CD_vals = CD0_arr[idx] + CD1_arr[idx] * CL_vals + CD2_arr[idx] * CL_vals**2
        ax_cl.plot(alpha_vals * r2d, CL_vals, color=cols_gradient(idx), label=lab)
        ax_cl.plot(raw_aero_data[idx]['CL'][:, 0], raw_aero_data[idx]['CL'][:, 1], 'x', color=cols[1])

        ax_cd.plot(alpha_vals * r2d, CD_vals, color=cols_gradient(idx))
        ax_cd.plot(raw_aero_data[idx]['CD'][:, 0], raw_aero_data[idx]['CD'][:, 1], 'x', color=cols[1])

        ax_ld.plot(alpha_vals * r2d, CL_vals / CD_vals, color=cols_gradient(idx), label=lab)

    ax_cl.legend()
    fig_curves.tight_layout()

    # Verify aero function fit
    mach_vals = np.linspace(0., 20., 100)
    CL0_vals = np.asarray(CL0_fun(mach_vals)).flatten()
    CLa_vals = np.asarray(CLa_fun(mach_vals)).flatten()
    CD0_vals = np.asarray(CD0_fun(mach_vals)).flatten()
    CD1_vals = np.asarray(CD1_fun(mach_vals)).flatten()
    CD2_vals = np.asarray(CD2_fun(mach_vals)).flatten()

    coeff_data = (CL0_arr, CLa_arr, CD0_arr, CD1_arr, CD2_arr)
    y_data = (CL0_vals, CLa_vals, CD0_vals, CD1_vals, CD2_vals)
    y_labels = (r'$C_{L,0}$ [-]', r'$C_{L,\alpha}$ [-]', r'$C_{D,0}$ [-]', r'$C_{D,1}$ [-]', r'$C_{D,2}$ [-]')

    fig_coeffs = plt.figure()
    axes = []
    for idx, coeff in enumerate(coeff_data):
        axes.append(fig_coeffs.add_subplot(5, 1, idx+1))
        ax = axes[idx]
        ax.plot(mach_vals, y_data[idx])
        ax.plot(machs, coeff, 'x')
        ax.grid()
        ax.set_xlabel('Mach')
        ax.set_ylabel(y_labels[idx])
    fig_coeffs.tight_layout()

    plt.show()
