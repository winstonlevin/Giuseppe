import numpy as np

from lookup_tables import cl_alpha_table, cd0_table, atm, dens_table, temp_table


# For consistency, use LUT values for speed of sound and density
def speed_of_sound(_h):
    _temperature = np.asarray(temp_table(_h)).flatten()
    _a = (_temperature * atm.gas_constant * atm.specific_heat_ratio) ** 0.5


def density(_h):
    _rho = np.asarray(dens_table(_h)).flatten()


