import os

import numpy as np
from scipy import interpolate, optimize
import matplotlib
from matplotlib import pyplot as plt
import casadi as ca

import giuseppe

os.chdir(os.path.dirname(__file__))  # Set diectory to current location
matplotlib.use('TkAgg', force=True)

# NOTE: All data for 0 <= M <= 0.8 is considered constant
lut_data = {
    'M': np.array((0, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8)),
    'CLa': np.array((3.44, 3.44, 3.44, 3.44, 3.44, 3.58, 4.44, 3.44, 3.01, 2.86, 2.44)),
    'CD0': np.array((0.013, 0.013, 0.013, 0.013, 0.013, 0.014, 0.031, 0.041, 0.039, 0.036, 0.035)),
    'eta': np.array((0.54, 0.54, 0.54, 0.54, 0.54, 0.75, 0.79, 0.78, 0.89, 0.93, 0.93))
}


# Fit:
# transonic rise represented by:
# Csup = Chyp + CT0 exp(-CT1(M-1))
#
# subsonic constancy represented by:
# Chat = 0.5*(Csup - Csub)*tanh(KT*(M-1)) + 0.5*(Csup + Csub)
#
# With the free parameters:
# CT0, CT1, Chyp, KT
#
# And the known parameters:
# Csub
M = ca.SX.sym('M')
Csub = ca.SX.sym('Csub')
Chyp = ca.SX.sym('Chyp')
KA = ca.SX.sym('KA')
CT0 = ca.SX.sym('CT0')
CT1 = ca.SX.sym('CT1')
yhat_sup = Chyp + CT0 * np.exp(-CT1*(M - 1.))
yhat = 0.5 * (yhat_sup + Csub) + 0.5 * (yhat_sup - Csub) * np.tanh(KA*(M-1.))

params = ca.vcat((Csub, Chyp, KA, CT0, CT1))
model = ca.Function('y', (M, params), (yhat,), ('M', 'p'), ('y',))


def wrap_model_params(_p):
    def _wrapped_fun(_x):
        return model(np.asarray(_x).reshape((1, -1)), _p).full().ravel()
    return _wrapped_fun


# Fit model to data (Newton descent)
def fit_model(_ydata):
    _c_sub = _ydata[0]

    _yhat = ca.substitute(model(lut_data['M'][None, :], params), Csub, _c_sub).T
    _err = _yhat - _ydata
    _obj = ca.dot(_err, _err)
    _grad = ca.jacobian(_obj, params[1:])
    _hess = ca.jacobian(_grad, params[1:])
    _obj_fun = ca.Function('f', (params[1:],), (_obj,))
    _grad_fun = ca.Function('g', (params[1:],), (_grad,))
    _hess_fun = ca.Function('J', (params[1:],), (_hess,))

    _chyp = _ydata[-1]
    idx_sonic = np.abs(lut_data['M'] - 1).argmin()
    _ct0 = 2*_ydata[idx_sonic] - _c_sub - _chyp
    _ka = np.arctanh(1-1E-1)/0.2
    _ct1 = (np.log((_ydata[idx_sonic+1:-1] - _chyp)/_ct0)/(1. - lut_data['M'][idx_sonic+1:-1])).mean()
    _z0 = np.array((_chyp, _ka, _ct0, _ct1))
    _sol = optimize.minimize(fun=lambda _z: _obj_fun(_z).full().ravel(), jac=lambda _z: _grad_fun(_z).full().ravel(), x0=_z0, hess=_hess_fun)

    return np.concatenate(((_c_sub,), _sol.x))


coeffs_CLa = fit_model(lut_data['CLa'])
coeffs_CD0 = fit_model(lut_data['CD0'])
coeffs_eta = fit_model(lut_data['eta'])

CLa_model = wrap_model_params(coeffs_CLa)
CD0_model = wrap_model_params(coeffs_CD0)
eta_model = wrap_model_params(coeffs_eta)

# Fit piecewise 6th order polynomial such that, at breakpoints (y(i), y(i+1)):
# (1) yhat(i)   = y(i)
# (2) dyhat(i)  = dy(i)  ~= [y(i-1) + y(i+1)] / [x(i-1) + x(i+1)]
# (3) d2yhat(i) = d2y(i) ~+ [dy(i-1) + dy(i+1)] / [x(i-1) + x(i+1)]
dm = np.diff(lut_data['M'])
idxf_subsonic = np.where(np.isclose(lut_data['M'], 0.8))[0][0]


# def fit_piecewise(_y):
#     _dy = _y[1:] - _y[:-1]
#     _ddy = np.diff(_dy)
#     _dydm = np.empty_like(_y)
#     _dydm[:idxf_subsonic] = 0.
#
#     # Central difference approximation for derivatives
#     _idx10 = idxf_subsonic+1
#     _idx1f = -1
#     _idx00 = idxf_subsonic - 1
#     _idx0f = -3
#     _dydm[idxf_subsonic:-1] = (_y[_idx10:_idx1f] - _y[_idx00:_idx0f]) / (m[_idx10:_idx1f] - m[_idx00:_idx0f])
#     _dydm[-1] =

# Fit data to Legendre polynomial
legendre_order = 7
fit_subsonic = True
use_lut_breakpoints = True
n_quad_data = 100
integrate_error = False

legendre_bases = [np.polynomial.Legendre.basis(deg=_i) for _i in range(legendre_order+1)]

if fit_subsonic:
    idx0 = 0
else:
    idx0 = np.where(np.isclose(lut_data['M'], 0.8))[0][0]
idces_m = np.arange(idx0, len(lut_data['M']), 1)

m_mean = 0.5*(lut_data['M'][idces_m[-1]] + lut_data['M'][idces_m[0]])
m_range = 0.5*(lut_data['M'][idces_m[-1]] - lut_data['M'][idces_m[0]])
if use_lut_breakpoints:
    mach_bp = lut_data['M'][idces_m]
    tau_m = (mach_bp - m_mean)/m_range
else:
    tau_m = giuseppe.utils.pseudospectral.lgl(n_quad_data+1)[0]
    mach_bp = m_mean + m_range * tau_m
nm = len(tau_m)

if integrate_error:
    int_matrix = giuseppe.utils.pseudospectral.integration_matrix(tau_m)
else:
    int_matrix = np.eye(nm)
legendre_matrix = np.empty(shape=(nm, legendre_order+1))
for idx, leg_poly in enumerate(legendre_bases):
    legendre_matrix[:, idx] = leg_poly(tau_m)
design_matrix = int_matrix @ legendre_matrix


def fit_data(_y):
    _y_int = int_matrix @ _y
    return np.linalg.lstsq(design_matrix, _y_int, rcond=None)


CLa_pchip_interp = interpolate.PchipInterpolator(lut_data['M'], lut_data['CLa'])
CD0_pchip_interp = interpolate.PchipInterpolator(lut_data['M'], lut_data['CD0'])
eta_pchip_interp = interpolate.PchipInterpolator(lut_data['M'], lut_data['eta'])

CLa_interp = np.polynomial.Legendre(fit_data(CLa_pchip_interp(mach_bp))[0])
CD0_interp = np.polynomial.Legendre(fit_data(CD0_pchip_interp(mach_bp))[0])
eta_interp = np.polynomial.Legendre(fit_data(eta_pchip_interp(mach_bp))[0])

# -------------------------------------------------------------------------------------------------------------------- #
# PLOTTING                                                                                                             #
# -------------------------------------------------------------------------------------------------------------------- #
tau_vals = np.linspace(-1, 1, 1000)
mach_vals = m_mean + m_range * tau_vals
CLa_vals = CLa_interp(tau_vals)
CD0_vals = CD0_interp(tau_vals)
eta_vals = eta_interp(tau_vals)

CLa_pchip_vals = CLa_pchip_interp(mach_vals)
CD0_pchip_vals = CD0_pchip_interp(mach_vals)
eta_pchip_vals = eta_pchip_interp(mach_vals)

CLa_model_vals = CLa_model(mach_vals)
CD0_model_vals = CD0_model(mach_vals)
eta_model_vals = eta_model(mach_vals)

cols = plt.rcParams['axes.prop_cycle'].by_key()['color']
fig, axes = plt.subplots(nrows=3, sharex=True)

ydata_model = (CLa_model_vals, CD0_model_vals, eta_model_vals)
ydata = (CLa_vals, CD0_vals, eta_vals)
ydata_pchip = (CLa_pchip_vals, CD0_pchip_vals, eta_pchip_vals)
ydata_fit = (CLa_pchip_interp(mach_bp), CD0_pchip_interp(mach_bp), eta_pchip_interp(mach_bp))
ydata_lut = (lut_data['CLa'], lut_data['CD0'], lut_data['eta'])
ylabels = (r'$C_{L,a}$', r'$C_{D,0}$', r'$\eta$')

for idx, ax in enumerate(axes):
    ax.grid(zorder=-1)
    ax.plot(mach_vals, ydata_pchip[idx], color=cols[2])
    ax.plot(mach_vals, ydata_model[idx], color=cols[0])
    # ax.plot(mach_vals, ydata[idx], color=cols[0])
    ax.scatter(mach_bp, ydata_fit[idx], marker='o', edgecolors=cols[1], facecolors='none')
    ax.plot(lut_data['M'], ydata_lut[idx], 'kx')
    ax.set_ylabel(ylabels[idx])
axes[-1].set_xlabel(r'$M$')

orig_str = 'LUT breakpoints' if use_lut_breakpoints else 'LG points'
int_str = 'integrated error' if integrate_error else 'error'
fig.suptitle(f'Interp [{legendre_order}th order poly, {int_str} @ {orig_str}]')
fig.tight_layout()
