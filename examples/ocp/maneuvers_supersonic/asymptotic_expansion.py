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


def find_root_depressed_cubic(x0, c1, c0, max_iter: int = 100, tol: float = 1e-6):
    # Find the real root for x**3 + c1x + c0 = 0
    x = x0

    for idx in range(max_iter):
        f = x0 ** 3 + c1 * x0 + c0
        fp = 3 * x0 ** 2 + c1

        x = x0 - f / fp

        if abs(x - x0) < tol:
            break

    return x


def glide_asymptotic_expansion(e, h, gam, h_glide, m, g, s_ref, eta):
    ad0, adl = get_accel_parameters(e, h, m, g, s_ref, eta)
    ad0_glide, adl_glide = get_accel_parameters(e, h_glide, m, g, s_ref, eta)

    radicand_gam = (ad0_glide - ad0 + adl_glide - adl) / adl

    # if any(radicand_gam < 0):
    #     warn('radicand_gam < 0! Setting gam1 = 0')
    #     gam1 = 0
    # else:
    #     gam1 = np.sign(h_glide - h) * np.arcsin(radicand_gam ** 0.5)

    c1 = -(2 * adl + ad0) / adl
    c0 = (ad0_glide + adl_glide) / adl
    cgam1 = find_root_depressed_cubic(1., c1, c0)

    if cgam1 > 1:
        cgam1 = 1
        warn('cos(gam1) > 1! Setting cos(gam1) = 1')
    elif cgam1 < 0:
        cgam1 = 0
        warn('cos(gam1) < 0! Setting cos(gam1) = 0')

    gam1 = np.arccos(cgam1)

    lam_h = -np.tan(gam1) + 2 * adl * np.sin(gam1) / (ad0_glide + adl_glide)

    radicand_u = (ad0_glide + adl_glide) / adl * (
            np.cos(gam1) - np.cos(gam) + lam_h * (np.sin(gam) - np.sin(gam1))
    ) + np.cos(gam)**2 - np.cos(gam1)**2

    if any(radicand_u < 0):
        warn('radicand_u < 0! Setting u = cos(gam)')
        load_factor = np.cos(gam)
    else:
        load_factor = np.cos(gam) + np.sign(gam1 - gam) * radicand_u ** 0.5

    return load_factor
