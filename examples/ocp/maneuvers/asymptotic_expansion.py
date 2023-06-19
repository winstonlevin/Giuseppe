from typing import Optional
from warnings import warn
import numpy as np

from lookup_tables import sped_table, dens_table, cl_alpha_table, cd0_table


def get_accel_parameters(e, h, m, g, s_ref, eta):
    weight = m * g
    v = (2 * (e - h * g)) ** 0.5

    qdyn = 0.5 * np.asarray(dens_table(h)).flatten() * v ** 2
    mach = v / np.asarray(sped_table(h)).flatten()

    ad0 = qdyn * s_ref * np.asarray(cd0_table(mach)).flatten() / weight
    adl = eta * weight / (qdyn * s_ref * np.asarray(cl_alpha_table(mach)).flatten())
    return ad0, adl


def find_root_depressed_cubic(x0, c1, c0, max_iter: int = 100, tol: float = 1e-6, x_tol: float = 1e-16):
    # Find the real root for x**3 + c1x + c0 = 0
    x = x0

    for idx in range(max_iter):
        f = x0 ** 3 + c1 * x0 + c0
        fp = 3 * x0 ** 2 + c1

        x = x0 - f / fp

        if abs(f) < tol:
            break
        elif abs(x - x0) < x_tol:
            break

    return x


def glide_asymptotic_expansion(e, h, gam, h_glide, m, g, s_ref, eta, gam0: Optional[float] = None):
    ad0, adl = get_accel_parameters(e, h, m, g, s_ref, eta)
    ad0_glide, adl_glide = get_accel_parameters(e, h_glide, m, g, s_ref, eta)

    c1 = -(2 * adl + ad0) / adl
    c0 = (ad0_glide + adl_glide) / adl

    if gam0 is None:
        gam0 = 0.

    cgam_sols = np.roots((1., 0., float(c1), float(c0)))
    cgam_sols_feasible = cgam_sols[np.where(
        np.logical_and(np.isreal(cgam_sols), np.logical_and(cgam_sols > -1., cgam_sols < 1.))
    )]

    if len(cgam_sols_feasible) > 0:
        cgam1 = cgam_sols_feasible[np.abs(cgam_sols_feasible - np.cos(gam0)).argmin()]
        gam1 = np.arccos(abs(cgam1)) * np.sign(h_glide - h) * np.sign(cgam1)
    else:
        warn('No feasible solutions! Setting gam1 = gam0')
        gam1 = gam0

    lam_h = -np.tan(gam1) + 2 * adl * np.sin(gam1) / (ad0_glide + adl_glide)

    radicand_u = (ad0_glide + adl_glide) / adl * (
            np.cos(gam1) - np.cos(gam) + lam_h * (np.sin(gam) - np.sin(gam1))
    ) + np.cos(gam)**2 - np.cos(gam1)**2

    if any(radicand_u < 0):
        warn('radicand_u < 0! Setting u = cos(gam)')
        load_factor = np.cos(gam)
    else:
        load_factor = np.cos(gam) + np.sign(gam1 - gam) * radicand_u ** 0.5

    return load_factor, gam1
