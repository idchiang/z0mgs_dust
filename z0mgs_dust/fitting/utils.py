import platform
import inspect
import numpy as np
import astropy.units as u
from astropy.constants import c, h, k_B

# Some constants
hkB_KHz = (h / k_B).to(u.K / u.Hz).value
B_const = 2e20 * (h / c**2).to(u.J * u.s**3 / u.m**2).value
c_ums = c.to(u.um / u.s).value
MBB_const = 0.00020884262122368297

# Parameters for each model
parameters = {'SE': ['dust.surface.density', 'dust.temperature', 'beta'],
              'FB': ['dust.surface.density', 'dust.temperature'],
              'BE': ['dust.surface.density', 'dust.temperature', 'beta2']}
axis_ids = {'SE': {'dust.surface.density': 1, 'dust.temperature': 0,
                   'beta': 2},
            'FB': {'dust.surface.density': 1, 'dust.temperature': 0},
            'BE': {'dust.surface.density': 1, 'dust.temperature': 0,
                   'beta2': 2}}

# Fitting grid parameters: min, max, step
grid_para = {'dust.surface.density': [-4.,  1., 0.025],
             'dust.temperature': [5., 50., 0.5],
             'beta': [-1.0, 4.0, 0.1],
             'beta2': [-1.0, 4.0, 0.1]}

# Fitting parameter properties
# 0: is log; 1: 1d array; 2: units[normal/log]
v_prop = {'dust.surface.density':
          [True, -1,
           [r'$\Sigma_d$ $[M_\odot {\rm pc}^{-2}]$',
            r'$\log(\Sigma_d$ $[M_\odot {\rm pc}^{-2}])$']],
          'dust.temperature': [False, -1, [r'$T_d$ [K]']],
          'beta': [False, -1, [r'$\beta$']],
          'beta2': [False, -1, [r'$\beta_2$']],
          'chi2': [False, -1, [r'$\chi^2$']]}
for p in grid_para.keys():
    v_prop[p][1] = np.arange(grid_para[p][0], grid_para[p][1], grid_para[p][2])

# Representative wavelengths for each instrument
band_wl = {'pacs70': 70.0, 'pacs100': 100.0, 'pacs160': 160.0,
           'spire250': 250.0, 'spire350': 350.0, 'spire500': 500.0,
           'mips24': 24.0, 'mips70': 70.0, 'mips160': 160.0}


# Blackbody SED. Faster after dropping units
def B_fast(T_d, freq):
    """ Return blackbody SED of temperature T(with unit) in MJy """
    """ Get T and freq w/o unit, assuming K and Hz """
    with np.errstate(over='ignore'):
        return B_const * freq**3 / (np.exp(hkB_KHz * freq / T_d) - 1)


# SE/FB models SED
def SEMBB(wavelength, Sigma_d, T_d, beta, kappa_160=10.10):
    freq = c_ums / wavelength
    return MBB_const * kappa_160 * (160.0 / wavelength)**beta * Sigma_d * \
        B_fast(T_d, freq)


# BE model SED
def BEMBB(wavelength, Sigma_d, T_d, beta, lambda_c, beta_2, kappa_160=20.73):
    """Return fitted SED in MJy"""
    freq = c_ums / wavelength
    ans = B_fast(T_d, freq) * MBB_const * kappa_160 * Sigma_d * 160.0**beta
    try:
        # Only allows 1-dim for all parameters. No error detection
        nwl = len(wavelength)
        del nwl
        small_mask = wavelength < lambda_c
        ans[small_mask] *= (1.0 / wavelength[small_mask])**beta
        ans[~small_mask] *= \
            (lambda_c**(beta_2 - beta)) * wavelength[~small_mask]**(-beta_2)
        return ans
    except TypeError:
        small_mask = wavelength < lambda_c
        ans[small_mask] *= (1.0 / wavelength)**beta
        ans[~small_mask] *= \
            (lambda_c[~small_mask]**(beta_2[~small_mask] - beta)) * \
            wavelength**(-beta_2[~small_mask])
        return ans


# Find the absolute path of this script
def abspath_and_sep():
    abs_path = inspect.getfile(inspect.currentframe())
    sep = '\\' if platform.system() == 'Windows' else '/'
    temp = abs_path.split(sep)
    abs_path = sep.join(temp[:-1]) + sep
    return abs_path, sep
