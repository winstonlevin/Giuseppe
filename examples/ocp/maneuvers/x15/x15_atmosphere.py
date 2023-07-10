import casadi as ca

from giuseppe.utils.examples import Atmosphere1976

mu = 1.4076539e16
Re = 20902900.
g0 = mu / Re ** 2

atm = Atmosphere1976(use_metric=False, earth_radius=Re, gravity=g0, boundary_thickness=1_000.)

h_SX = ca.SX.sym('h')

temp_expr, pres_expr, dens_expr = atm.get_ca_atm_expr(h_SX)
sped_expr = atm.get_ca_speed_of_sound_expr(h_SX)

temp_fun = ca.Function('T', (h_SX,), (temp_expr,), ('h',), ('T',))
pres_fun = ca.Function('P', (h_SX,), (pres_expr,), ('h',), ('P',))
dens_fun = ca.Function('rho', (h_SX,), (dens_expr,), ('h',), ('rho',))
sped_fun = ca.Function('a', (h_SX,), (sped_expr,), ('h',), ('a',))
