import numpy as np
import casadi as ca

from giuseppe.utils.examples import Atmosphere1976, create_buffered_linear_interpolator, \
    create_buffered_2d_linear_interpolator

mu = 1.4076539e16
Re = 20902900.
g0 = mu / Re ** 2

atm = Atmosphere1976(use_metric=False, earth_radius=Re, gravity=g0, boundary_thickness=1_000.)
lut_data = {
    'M': np.array((0., 0.4, 0.8, 1.2, 1.6, 2., 2.4, 2.8, 3.2)),
    'h': 1e3 * np.array((-2., 0., 5., 15., 25., 35., 45., 55., 65., 75., 85., 95., 105.)),
    'CLalpha': np.array((2.240, 2.325, 2.350, 2.290, 2.160, 1.950, 1.700, 1.435, 1.250)),
    'CD0': np.array((0.0065, 0.0055, 0.0060, 0.0118, 0.0110, 0.00860, 0.0074, 0.0069, 0.0068)),
    'T': 1e3 * np.array((
        (23.3, 23.3, 20.6, 15.4, 9.9, 5.8, 2.9, 1.3, 0.7, 0.3, 0.1, 0.1, 0.),
        (22.8, 22.8, 19.8, 14.4, 9.9, 6.2, 3.4, 1.7, 1.0, 0.5, 0.3, 0.1, 0.1),
        (24.5, 24.5, 22.0, 16.5, 12.0, 7.9, 4.9, 2.8, 1.6, 0.9, 0.5, 0.3, 0.2),
        (29.4, 29.4, 27.3, 21.0, 15.8, 11.4, 7.2, 3.8, 2.7, 1.6, 0.9, 0.6, 0.4),
        (29.7, 29.7, 29.0, 27.5, 21.8, 15.7, 10.5, 6.5, 3.8, 2.3, 1.4, 0.8, 0.5),
        (29.9, 29.9, 29.4, 28.4, 26.6, 21.2, 14.0, 8.7, 5.1, 3.3, 1.9, 1.0, 0.5),
        (29.9, 29.9, 29.2, 28.4, 27.1, 25.6, 17.2, 10.7, 6.5, 4.1, 2.3, 1.2, 0.5),
        (29.8, 29.8, 29.1, 28.2, 26.8, 25.6, 20.0, 12.2, 7.6, 4.7, 2.8, 1.4, 0.5),
        (29.7, 29.7, 28.9, 27.5, 26.1, 24.9, 20.3, 13.0, 8.0, 4.9, 2.8, 1.4, 0.5),
    ))
}

lut_data['Theta'] = np.asarray([atm.temperature(alt) for alt in lut_data['h']])
lut_data['a'] = np.asarray([atm.speed_of_sound(alt) for alt in lut_data['h']])
lut_data['rho'] = np.asarray([atm.density(alt) for alt in lut_data['h']])

# Create Look-up Tables
# thrust_table = ca.interpolant('T', 'bspline', (lut_data['M'], lut_data['h']), lut_data['T'].ravel(order='F'))
# cl_alpha_table = ca.interpolant('CLalpha', 'bspline', (lut_data['M'],), lut_data['CLalpha'])
# cd0_table = ca.interpolant('CD0', 'bspline', (lut_data['M'],), lut_data['CD0'])
temp_table = ca.interpolant('T', 'bspline', (lut_data['h'],), lut_data['Theta'])
sped_table = ca.interpolant('a', 'bspline', (lut_data['h'],), lut_data['a'])
dens_table = ca.interpolant('rho', 'bspline', (lut_data['h'],), lut_data['rho'])

# Create conditional CasADi functions
h_SX = ca.SX.sym('h', 1)
sped_expr = atm.get_ca_speed_of_sound_expr(h_SX)
temp_expr, pres_expr, dens_expr = atm.get_ca_atm_expr(h_SX)

sped_fun = ca.Function('a', (h_SX,), (sped_expr,), ('h',), ('a',))
temp_fun = ca.Function('T', (h_SX,), (temp_expr,), ('h',), ('T',))
pres_fun = ca.Function('P', (h_SX,), (pres_expr,), ('h',), ('P',))
dens_fun = ca.Function('rho', (h_SX,), (dens_expr,), ('h',), ('rho',))

# Create Conditional Aero Functions
mach_bl = 0.2
h_bl = 1e3
mach_SX = ca.SX.sym('M', 1)
thrust_expr = create_buffered_2d_linear_interpolator(lut_data['M'], lut_data['h'], lut_data['T'],
                                                     [mach_SX, h_SX], (mach_bl, h_bl))
cl_alpha_expr = create_buffered_linear_interpolator(lut_data['M'], lut_data['CLalpha'],
                                                    mach_SX, mach_bl)
cd0_expr = create_buffered_linear_interpolator(lut_data['M'], lut_data['CD0'],
                                               mach_SX, mach_bl)

thrust_fun = ca.Function('T', (mach_SX, h_SX), (thrust_expr,), ('M', 'h'), ('T',))
cl_alpha_fun = ca.Function('CLa', (mach_SX,), (cl_alpha_expr,), ('M',), ('CLa',))
cd0_fun = ca.Function('CD0', (mach_SX,), (cd0_expr,), ('M',), ('CD0',))

if __name__ == '__main__':
    from matplotlib import cm, pyplot as plt

    n_vals = 1_000
    h_vals = np.linspace(lut_data['h'][0] - 1_000., lut_data['h'][-1] + 1_000., n_vals)
    M_vals = np.linspace(lut_data['M'][0] - 0.25, lut_data['M'][-1] + 0.25, n_vals)

    rho_vals = np.empty(h_vals.shape)
    theta_vals = np.empty(h_vals.shape)
    for idx, h in enumerate(h_vals):
        theta_vals[idx], _, rho_vals[idx] = atm.atm_data(h)

    a_vals = (theta_vals * atm.gas_constant * atm.specific_heat_ratio) ** 0.5

    M_grid, h_grid = np.meshgrid(M_vals, h_vals)
    M_bp_grid, h_bp_grid = np.meshgrid(lut_data['M'], lut_data['h'])
    thrust_vals = np.empty(M_grid.shape)
    for idx in range(len(M_grid[:, 0])):
        thrust_vals[idx, :] = np.asarray(thrust_fun(M_grid[idx, :], h_grid[idx, :])).flatten()

    fig_aero = plt.figure()

    for idx, (fun, ylab) in enumerate(zip((cl_alpha_fun, cd0_fun), ('CLalpha', 'CD0'))):
        ax = fig_aero.add_subplot(2, 1, idx + 1)
        ax.plot(M_vals, fun(M_vals))
        ax.plot(lut_data['M'], fun(lut_data['M']), 'kx')
        ax.grid()
        ax.set_xlabel('Mach')
        ax.set_ylabel(ylab)

    fig_aero.tight_layout()

    fig_atm = plt.figure()
    axes_atm = []

    for idx, (fun, ylab, ref) in enumerate(zip((temp_table, dens_table, sped_table),
                                               ('Temperature [R]', 'Density [slug/ft3]', 'Speed of Sound [ft/s]'),
                                               (theta_vals, rho_vals, a_vals))):
        axes_atm.append(fig_atm.add_subplot(3, 1, idx + 1))
        ax = axes_atm[-1]
        ax.plot(h_vals, fun(h_vals))
        ax.plot(h_vals, ref, zorder=0)
        ax.plot(lut_data['h'], fun(lut_data['h']), 'kx')
        ax.set_xlabel('Altitude [ft]')
        ax.set_ylabel(ylab)

    fig_thrust, ax = plt.subplots(subplot_kw=dict(projection='3d'))
    ax.plot_surface(M_grid, h_grid, thrust_vals, alpha=0.25, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.scatter(
        M_bp_grid.flatten(),
        h_bp_grid.flatten(),
        lut_data['T'].T.flatten()
    )

    plt.show()
