import numpy
import numpy as np
from matplotlib import pyplot as plt

import giuseppe

n_col = 6


def _get_weight(_tau_col, _tau_new):
    _tau_all = np.unique(np.append(_tau_col, _tau_new))
    _idx_new = np.where(np.isclose(_tau_all, _tau_new))[0][0]
    _weights = giuseppe.utils.pseudospectral.integration_matrix(_tau_all, np.array((1.,)))
    return _weights[0, _idx_new]


col_str = 'lgl'
if col_str == 'lgl':
    tau_col, weights_col = giuseppe.utils.pseudospectral.lgl(n_col)
elif col_str == 'lg':
    tau_col, weights_col = giuseppe.utils.pseudospectral.lg(n_col+1)
    tau_col = tau_col[1:]
elif col_str == 'lgr':
    tau_col, weights_col = giuseppe.utils.pseudospectral.lgr(n_col)
else:
    raise ValueError(f'col_str="{col_str}" is not implemented!')


tau_new_vals = np.linspace(-1.5, 1.5, 1000)
weight_new_vals = np.array([_get_weight(tau_col, _tau_new) for _tau_new in tau_new_vals])

# Plotting ----------------------------------------------------------------------------------------------------------- #
fig, ax = plt.subplots()

ax.grid(zorder=-1)
ax.plot(tau_new_vals, weight_new_vals)
ax.set_xlabel(r'$\tau_i$')
ax.set_ylabel(r'$w_i$')
ax.set_title(f'Weight of Extra Col. Pt. [{col_str.upper()} - {n_col} Pts.]')
