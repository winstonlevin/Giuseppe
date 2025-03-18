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
    _dyt = _ydata[idx_sonic+1:-1] - _chyp
    _mt = lut_data['M'][idx_sonic+1:-1]
    idces_valid = np.sign(_dyt) == np.sign(_ct0)
    if np.any(idces_valid):
        _ct1 = (np.log(_dyt[idces_valid]/_ct0)/(1. - _mt[idces_valid])).mean()
    else:
        _ct1 = 0.
    _z0 = np.array((_chyp, _ka, _ct0, _ct1))
    _sol = optimize.minimize(fun=lambda _z: _obj_fun(_z).full().ravel(), jac=lambda _z: _grad_fun(_z).full().ravel(), x0=_z0, hess=_hess_fun)

    return np.concatenate(((_c_sub,), _sol.x))


coeffs_CLa = fit_model(lut_data['CLa'])
coeffs_CD0 = fit_model(lut_data['CD0'])
coeffs_etaCLa = fit_model(lut_data['eta'] * lut_data['CLa'])

CLa_model = wrap_model_params(coeffs_CLa)
CD0_model = wrap_model_params(coeffs_CD0)
etaCLa_model = wrap_model_params(coeffs_etaCLa)

# -------------------------------------------------------------------------------------------------------------------- #
# PLOTTING                                                                                                             #
# -------------------------------------------------------------------------------------------------------------------- #
mach_vals = np.linspace(lut_data['M'][0], lut_data['M'][-1], 1000)

CLa_model_vals = CLa_model(mach_vals)
CD0_model_vals = CD0_model(mach_vals)
etaCLa_model_vals = etaCLa_model(mach_vals)

cols = plt.rcParams['axes.prop_cycle'].by_key()['color']
fig, axes = plt.subplots(nrows=3, sharex=True)

ydata_model = (CLa_model_vals, CD0_model_vals, etaCLa_model_vals)
ydata_lut = (lut_data['CLa'], lut_data['CD0'], lut_data['eta']*lut_data['CLa'])
ylabels = (r'$C_{L,a}$', r'$C_{D,0}$', r'${\eta}C_{L,a}$')

for idx, ax in enumerate(axes):
    ax.grid(zorder=-1)
    ax.plot(mach_vals, ydata_model[idx], color=cols[0])
    ax.plot(lut_data['M'], ydata_lut[idx], 'kx')
    ax.set_ylabel(ylabels[idx])
axes[-1].set_xlabel(r'$M$')
fig.tight_layout()
