import os
import time
import h5py
import multiprocessing as mp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import astropy.units as u
from matplotlib.backends.backend_pdf import PdfPages
from astropy.io import fits
# from astropy.wcs import WCS
# from astropy.coordinates import Angle
from astropy.constants import c, N_A
# from corner import corner
from sklearn.linear_model import LinearRegression
# from .idc_voronoi import voronoi_m
from .idc_functions import SEMBB, BEMBB, WD, PowerLaw, B_fast
from .idc_functions import best_fit_and_error, normalize_pdf, save_fits_gz
from .idc_fitting_old import kappa_calibration as kc_old
from .z0mg_RSRF import z0mg_RSRF
plt.ioff()

# Model properties
ndims = {'SE': 3, 'FB': 2, 'WD': 3, 'BE': 3, 'PL': 4}
# Parameters for each model
parameters = {'SE': ['dust.surface.density', 'dust.temperature', 'beta'],
              'FB': ['dust.surface.density', 'dust.temperature'],
              'BE': ['dust.surface.density', 'dust.temperature', 'beta2'],
              'WD': ['dust.surface.density', 'dust.temperature',
                     'warm.dust.fraction'],
              'PL': ['dust.surface.density', 'alpha', 'gamma', 'logUmin']}
axis_ids = {'SE': {'dust.surface.density': 1, 'dust.temperature': 0,
                   'beta': 2},
            'FB': {'dust.surface.density': 1, 'dust.temperature': 0},
            'BE': {'dust.surface.density': 1, 'dust.temperature': 0,
                   'beta2': 2},
            'WD': {'dust.surface.density': 1, 'dust.temperature': 0,
                   'warm.dust.fraction': 2},
            'PL': {'dust.surface.density': 1, 'alpha': 0, 'gamma': 2,
                   'logUmin': 3}}

# Grid parameters: min, max, step
"""
grid_para = {'dust.surface.density': [-4.,  1., 0.025],
             'dust.temperature': [5., 50., 0.5],
             'beta': [-1.0, 4.0, 0.1],
             'beta2': [-1.0, 4.0, 0.1],
             'warm.dust.fraction': [0.0, 0.05, 0.002],
             'alpha': [1.1, 3.0, 0.1],  # Remember to avoid alpha==1
             'gamma': [-4, 0, 0.2],
             'logUmin': [-2, 1.5, 0.1]}
"""
grid_para = {'dust.surface.density': [-4.,  1., 0.025],
             'dust.temperature': [5., 50., 0.5],
             'beta': [-1.0, 4.0, 0.1],
             'beta2': [-1.0, 4.0, 0.25],
             'warm.dust.fraction': [0.0, 0.05, 0.002],
             'alpha': [1.1, 5.1, 0.2],  # Remember to avoid alpha==1
             'gamma': [-3, 0, 0.2],
             'logUmin': [-1.5, 1.5, 0.2]}
# Parameter properties
# 0: is log; 1: 1d array; 2: units[normal/log]
v_prop = {'dust.surface.density':
          [True, -1,
           [r'$\Sigma_d$ $[M_\odot {\rm pc}^{-2}]$',
            r'$\log(\Sigma_d$ $[M_\odot {\rm pc}^{-2}])$']],
          'dust.temperature': [False, -1, [r'$T_d$ [K]']],
          'beta': [False, -1, [r'$\beta$']],
          'beta2': [False, -1, [r'$\beta_2$']],
          'warm.dust.fraction': [False, -1, [r'$f_w$']],
          'alpha': [False, -1, [r'$\alpha$']],
          'gamma': [True, -1, [r'$\gamma$', r'$\log(\gamma)$']],
          'logUmin': [False, -1, [r'$\log(U)_{min}$']],
          'chi2': [False, -1, [r'$\chi^2$']]}
for p in grid_para.keys():
    v_prop[p][1] = np.arange(grid_para[p][0], grid_para[p][1], grid_para[p][2])

# Band and instrument properties
all_instr = ['pacs', 'spire', 'mips']
band_wl = {'pacs70': 70.0, 'pacs100': 100.0, 'pacs160': 160.0,
           'spire250': 250.0, 'spire350': 350.0, 'spire500': 500.0,
           'mips24': 24.0, 'mips70': 70.0, 'mips160': 160.0}
band_cap = {'pacs70': 'PACS_70', 'pacs100': 'PACS_100', 'pacs160': 'PACS_160',
            'spire250': 'SPIRE_250', 'spire350': 'SPIRE_350',
            'spire500': 'SPIRE_500',
            'mips24': 'MIPS_24', 'mips70': 'MIPS_70', 'mips160': 'MIPS_160'}
band_instr = {'pacs70': 'pacs', 'pacs100': 'pacs', 'pacs160': 'pacs',
              'spire250': 'spire', 'spire350': 'spire', 'spire500': 'spire',
              'mips24': 'mips', 'mips70': 'mips', 'mips160': 'mips'}
# For calibration error
# MIPS: from Spitzer cookbook
cau = {'pacs': 10.0 / 100.0, 'spire': 8.0 / 100.0, 'mips': 2.0 / 100.0}
cru = {'pacs70': 2.0 / 100, 'pacs100': 2.0 / 100, 'pacs160': 2.0 / 100,
       'spire250': 1.5 / 100, 'spire350': 1.5 / 100, 'spire500': 1.5 / 100,
       'mips24': 4.0 / 100, 'mips70': 7.0 / 100, 'mips160': 12.0 / 100}
# For integrals
FWHM = {'SPIRE_500': 36.09, 'SPIRE_350': 24.88, 'SPIRE_250': 18.15,
        'Gauss_25': 25, 'PACS_160': 11.18, 'PACS_100': 7.04,
        'HERACLES': 13}


def diskmask_UTOMO18(name, ress,
                     bands=['pacs100', 'pacs160', 'spire250', 'spire350',
                            'spire500'],
                     datapath='data/UTOMO18_dust/',
                     projectpath='Projects/UTOMO18/'):
    for res in ress:
        masks = []
        respath = datapath + name + '/' + res + '/'
        fns = os.listdir(respath)
        for band in bands:
            for fn in fns:
                temp = fn.split('_')
                if len(temp) < 4:
                    continue
                elif temp[-4] == band:
                    if temp[-1] == 'mask.fits':
                        temp, hdr = fits.getdata(respath + fn, header=True)
                        masks.append(temp.astype(bool))
        assert len(masks) == len(bands)
        diskmask = np.all(masks, axis=0)
        fn = respath + name + '_diskmask.fits'
        fits.writeto(fn, diskmask.astype(int), hdr, overwrite=True)


def bkgcov_UTOMO18(name, res_good, res_all,
                   bands=['pacs100', 'pacs160', 'spire250', 'spire350',
                          'spire500'],
                   datapath='data/UTOMO18_dust/',
                   projectpath='Projects/UTOMO18/'):
    print('Generating bkgcov for', name)
    res_good_num = [int(r.strip('res_').strip('pc')) for r in res_good]
    res_all_num = [int(r.strip('res_').strip('pc')) for r in res_all]
    nwl = len(bands)
    bkgcov_good = []
    bands_wl = ['100', '160', '250', '350', '500']
    for res in res_good:
        print('--good res:', res)
        seds = []
        respath = datapath + name + '/' + res + '/'
        mask_fn = respath + name + '_diskmask.fits'
        if not os.path.isfile(mask_fn):
            diskmask_UTOMO18(name, res_good)
        diskmask = fits.getdata(mask_fn).astype(bool)
        fns = os.listdir(respath)
        for band in bands:
            for fn in fns:
                temp = fn.split('_')
                if len(temp) < 4:
                    continue
                elif temp[-4] == band:
                    if temp[-1] != 'mask.fits':
                        seds.append(fits.getdata(respath + fn))
        assert len(seds) == nwl
        seds = np.array(seds)
        non_nanmask = np.all(np.isfinite(seds), axis=0)
        bkgmask = (~diskmask) * non_nanmask
        # implement outlier rejection
        outliermask = np.zeros_like(bkgmask, dtype=bool)
        for i in range(nwl):
            AD = np.abs(seds[i] - np.median(seds[i][bkgmask]))
            MAD = np.median(AD[bkgmask])
            with np.errstate(invalid='ignore'):
                outliermask += AD > 3 * MAD
        bkgmask = bkgmask * (~outliermask)
        # assert np.sum(bkgmask) > 10
        bkgcov = np.cov(seds[:, bkgmask])
        if name == 'm31':
            bkgcov[0, 3] = 0
            bkgcov[0, 4] = 0
            bkgcov[3, 0] = 0
            bkgcov[4, 0] = 0
        if name == 'm33':
            for i in range(5):
                for j in range(5):
                    if i != j:
                        bkgcov[i, j] = 0
        bkgcov_good.append(bkgcov)
        fn = respath + name + '_bkgcov.fits'
        fits.writeto(fn, bkgcov, overwrite=True)
    bkgcov_check = []
    for res in res_all[:-1]:
        print('--check res:', res)
        seds = []
        respath = datapath + name + '/' + res + '/'
        mask_fn = respath + name + '_diskmask.fits'
        if not os.path.isfile(mask_fn):
            diskmask_UTOMO18(name, res_good)
        diskmask = fits.getdata(mask_fn).astype(bool)
        fns = os.listdir(respath)
        for band in bands:
            for fn in fns:
                temp = fn.split('_')
                if len(temp) < 4:
                    continue
                elif temp[-4] == band:
                    if temp[-1] != 'mask.fits':
                        seds.append(fits.getdata(respath + fn))
        assert len(seds) == nwl
        seds = np.array(seds)
        non_nanmask = np.all(np.isfinite(seds), axis=0)
        bkgmask = (~diskmask) * non_nanmask
        # implement outlier rejection
        outliermask = np.zeros_like(bkgmask, dtype=bool)
        for i in range(nwl):
            AD = np.abs(seds[i] - np.median(seds[i][bkgmask]))
            MAD = np.median(AD[bkgmask])
            with np.errstate(invalid='ignore'):
                outliermask += AD > 3 * MAD
        bkgmask = bkgmask * (~outliermask)
        # assert np.sum(bkgmask) > 10
        bkgcov = np.cov(seds[:, bkgmask])
        bkgcov_check.append(bkgcov)
    bkgcov_check = np.array(bkgcov_check)
    # build linear models
    bkgcov_good = np.array(bkgcov_good)
    models = np.empty([nwl, nwl], dtype=object)
    for i in range(nwl):
        for j in range(nwl):
            models[i, j] = LinearRegression(fit_intercept=True)
            if (name == 'm31') and \
                    ([i, j] in ([0, 3], [0, 4], [3, 0], [4, 0])):
                pass
            elif (name == 'm33') and (i != j):
                pass
            else:
                models[i, j].fit(np.log10(res_good_num).reshape([-1, 1]),
                                 np.log10(np.abs(bkgcov_good[:, i, j]))
                                 .reshape([-1, 1]))
    # fit the rest (or maybe all)
    bkgcov_fit = []
    for r in range(len(res_all)):
        res = res_all[r]
        print('----fit res:', res)
        res_num = res_all_num[r]
        bkgcov = np.zeros([nwl, nwl])
        for i in range(nwl):
            for j in range(nwl):
                if (name == 'm31') and \
                        ([i, j] in ([0, 3], [0, 4], [3, 0], [4, 0])):
                    pass
                elif (name == 'm33') and (i != j):
                    pass
                else:
                    bkgcov[i, j] = \
                        models[i, j].predict(np.log10(res_num))[0, 0]
                    bkgcov[i, j] = 10**bkgcov[i, j]
        bkgcov_fit.append(bkgcov)
        if res not in res_good:
            respath = datapath + name + '/' + res + '/'
            fn = respath + name + '_bkgcov.fits'
            fits.writeto(fn, bkgcov, overwrite=True)
    bkgcov_fit = np.array(bkgcov_fit)
    # plot true and fitting
    ylims = {'lmc': (5*10**(-4), 10**(1)),
             'smc': (2*10**(-7), 7*10**(0)),
             'm31': (10**(-5), 1.5*10**(1)),
             'm33': (10**(-3), 10**(1))}
    fig, ax = plt.subplots(nwl, nwl, figsize=(10, 10))
    for i in range(nwl):
        for j in range(nwl):
            ax[i, j].loglog(res_all_num, bkgcov_fit[:, i, j],
                            marker='+', ms=10, color='b', lw=0.5)
            ax[i, j].scatter(res_all_num[:-1], bkgcov_check[:, i, j],
                             s=20, marker='x', color='r')
            ax[i, j].set_ylim(ylims[name])
            ax[i, j].plot([res_good_num[-1]] * 2, ax[i, j].get_ylim(), 'k--',
                          alpha=0.3)
            ax[i, j].set_title(bands_wl[i] + '-' + bands_wl[j], color='k',
                               x=0.95, y=0.8, ha='right')
            if i == 4:
                ax[i, j].set_xlabel('resolution (pc)')
            else:
                ax[i, j].set_xticklabels([])
            if j != 0:
                ax[i, j].set_yticklabels([])
    fig.tight_layout()
    fig.savefig('output/' + name + '.png')
    plt.close('all')


def fit_dust_density(name, beta_f, bands,
                     lambdac_f=300.0, method_abbr='FB', del_model=False,
                     fake=False, nop=5, targetSN=5, Voronoi=False,
                     project_name='UTOMO18', save_pdfs=True, rand_cube=False,
                     observe_fns=[], mask_fn='', subdir=None,
                     notes='', galactic_integrated=False,
                     better_bkgcov=None, res_arcsec=None,
                     import_beta=False, beta_in=None, input_avg_SED=False,
                     avg_SED=[]):
    assert len(observe_fns) == len(bands)
    randcubesize = 100
    #
    nwl = len(bands)
    diskmask, hdr = fits.getdata(mask_fn, header=True)
    diskmask = diskmask.astype(bool)
    list_shape = list(diskmask.shape)
    sed = np.empty(list_shape + [nwl])
    for i in range(nwl):
        sed[:, :, i] = fits.getdata(observe_fns[i], header=False)
    non_nanmask = np.all(np.isfinite(sed), axis=-1)
    diskmask = diskmask * non_nanmask
    bkgmask = (~diskmask) * non_nanmask
    # method_abbr: SE, FB, BE, WD, PL
    #
    # Reading wavelength #
    #
    wl = np.array([band_wl[b] for b in bands])
    # Define cali_mat2
    cali_mat2 = np.zeros([nwl, nwl])
    for instr in all_instr:
        instr_bands = [bi for bi in range(nwl) if
                       band_instr[bands[bi]] == instr]
        for bi in instr_bands:
            cali_mat2[bi, bi] += cru[bands[bi]]
            for bj in instr_bands:
                cali_mat2[bi, bj] += cau[instr]
    cali_mat2 = cali_mat2**2
    #
    # Reading calibration #
    #
    if import_beta:
        beta_in = np.round_(beta_in, 2)
        beta_unique = np.unique(beta_in[diskmask])
        assert len(beta_unique) < 100  # Please not too many...
        kappa160s = {}
        for b in beta_unique:
            fn = 'hdf5_MBBDust/Calibration_' + str(round(b, 2)) + '.h5'
            try:
                with h5py.File(fn, 'r') as hf:
                    grp = hf[method_abbr]
                    kappa160s[b] = grp['kappa160'][()]
            except (KeyError, NameError, OSError):
                print('This method is not calibrated yet!!',
                      'Starting calibration...')
                kc_old(method_abbr, beta_f=b, lambdac_f=lambdac_f,
                       nop=nop)
                with h5py.File(fn, 'r') as hf:
                    grp = hf[method_abbr]
                    kappa160s[b] = grp['kappa160'][()]
    else:
        fn = 'hdf5_MBBDust/Calibration_' + str(round(beta_f, 2)) + '.h5'
        try:
            with h5py.File(fn, 'r') as hf:
                grp = hf[method_abbr]
                kappa160 = grp['kappa160'][()]
        except (KeyError, NameError, OSError):
            print('This method is not calibrated yet!!',
                  'Starting calibration...')
            kappa_calibration(method_abbr, beta_f=beta_f, lambdac_f=lambdac_f,
                              nop=nop)
            with h5py.File(fn, 'r') as hf:
                grp = hf[method_abbr]
                kappa160 = grp['kappa160'][()]
    #
    """ Read HERSCHEL SED and diskmask """
    #
    betastr = 'free' if method_abbr == 'SE' else str(round(beta_f, 2))
    longname = name + ' ' + method_abbr + '.beta=' + betastr + ' ' + notes
    #
    print('################################################')
    print(longname + ' fitting (' + time.ctime() + ')')
    print('################################################')
    # Dust density in Solar Mass / pc^2
    # kappa_lambda in cm^2 / g
    # SED in MJy / sr
    if better_bkgcov is None:
        # implement outlier rejection
        outliermask = np.zeros_like(bkgmask, dtype=bool)
        for i in range(nwl):
            AD = np.abs(sed[:, :, i] - np.median(sed[bkgmask][i]))
            MAD = np.median(AD[bkgmask])
            with np.errstate(invalid='ignore'):
                outliermask += AD > 3 * MAD
        bkgmask = bkgmask * (~outliermask)
        new_bkgmask = bkgmask * (~outliermask)
        # assert np.sum(bkgmask) > 10
        bkgcov = np.cov(sed[new_bkgmask].T)
    else:
        bkgcov = better_bkgcov
    #
    if galactic_integrated:
        # bkgcov = better_bkgcov
        bkgcov = np.zeros([nwl, nwl])  # Power law approximation
        if input_avg_SED:
            sed_avg = np.array(avg_SED)
        else:
            sed_avg = np.array([np.mean(sed[:, :, i][diskmask]) for i in
                                range(nwl)])
        sed = sed_avg.reshape(1, 1, nwl)
        diskmask = np.ones([1, 1]).astype(int)
        list_shape = [1, 1]
        """
        spire500_beamsize = 1804.31
        ctr = np.array(list_shape) // 2
        #
        num_pix_inte = np.sum(diskmask)
        sed_avg = np.array([np.mean(sed[:, :, i][diskmask]) for i in
                            range(nwl)])
        sed = sed_avg.reshape(1, 1, nwl)
        diskmask = np.ones([1, 1]).astype(int)
        list_shape = [1, 1]
        #
        ps = np.zeros(2)
        w = WCS(hdr, naxis=2)
        xs, ys = \
            w.wcs_pix2world([ctr[0] - 1, ctr[0] + 1, ctr[0], ctr[0]],
                            [ctr[1], ctr[1], ctr[1] - 1, ctr[1] + 1], 1)
        ps[0] = np.abs(xs[0] - xs[1]) / 2 * \
            np.cos(Angle((ys[0] + ys[1]) * u.deg).rad / 2)
        ps[1] = np.abs(ys[3] - ys[2]) / 2
        ps *= u.degree.to(u.arcsec)
        ps = np.mean(ps)
        resolution_element = np.pi * (FWHM['SPIRE_500'] / 2)**2 / ps**2
        num_res = num_pix_inte / resolution_element
        if num_res > 1:
            bkgcov /= num_res
        """
    #
    """ Voronoi binning """
    #
    if Voronoi:
        assert False
    #
    """ Build or load SED models """
    # Should be a for loop or something like that here
    if import_beta:
        modelss = {}
        for be in beta_unique:
            models = []
            if del_model:
                for b in bands:
                    fn = 'models/' + b + '_' + method_abbr + '.beta=' + \
                        str(round(be, 2)) + '.fits.gz'
                    if os.path.isfile(fn):
                        os.remove(fn)
            for b in bands:
                fn = 'models/' + b + '_' + method_abbr + '.beta=' + \
                    str(round(be, 2)) + '.fits.gz'
                if not os.path.isfile(fn):
                    if method_abbr in ['SE']:
                        filelist = os.listdir('models')
                        new_fn = ''
                        for f in filelist:
                            temp = f.split('_')
                            if len(temp) > 1:
                                if (temp[0] == b) and \
                                        (temp[1][:2] == method_abbr):
                                    new_fn = f
                                    break
                        if new_fn == '':
                            models_creation(method_abbr, be, lambdac_f,
                                            band_instr[b], kappa160s[be], nop)
                        else:
                            fn = new_fn
                    else:
                        models_creation(method_abbr, be, lambdac_f,
                                        band_instr[b], kappa160s[be], nop)
                models.append(fits.getdata(fn))
            models = np.array(models)
            models = np.moveaxis(models, 0, -1)
            modelss[be] = models
    else:
        models = []
        if del_model:
            for b in bands:
                fn = 'models/' + b + '_' + method_abbr + '.beta=' + \
                    str(round(beta_f, 2)) + '.fits.gz'
                if os.path.isfile(fn):
                    os.remove(fn)
        for b in bands:
            fn = 'models/' + b + '_' + method_abbr + '.beta=' + \
                str(round(beta_f, 2)) + '.fits.gz'
            if not os.path.isfile(fn):
                if method_abbr in ['SE']:
                    filelist = os.listdir('models')
                    new_fn = ''
                    for f in filelist:
                        temp = f.split('_')
                        if len(temp) > 1:
                            if (temp[0] == b) and (temp[1][:2] == method_abbr):
                                new_fn = f
                                break
                    if new_fn == '':
                        models_creation(method_abbr, beta_f, lambdac_f,
                                        band_instr[b], kappa160, nop)
                    else:
                        fn = new_fn
                else:
                    models_creation(method_abbr, beta_f, lambdac_f,
                                    band_instr[b], kappa160, nop)
            models.append(fits.getdata(fn))
        models = np.array(models)
        models = np.moveaxis(models, 0, -1)
    #
    """ Real fitting starts """
    #
    axis_id = axis_ids[method_abbr]
    #
    temp_log = np.full([3] + list_shape, np.nan)
    temp_linear = np.full([2] + list_shape, np.nan)
    recovered_sed = np.full([nwl] + list_shape, np.nan)
    chi2_map = np.full(sed.shape[:2], np.nan)
    v_map, v_min, v_max = {}, {}, {}
    for p in parameters[method_abbr]:
        v_map[p] = np.full_like(temp_log, np.nan) if v_prop[p][0] else \
            np.full_like(temp_linear, np.nan)
        v_min[p] = np.full(list_shape, np.nan)
        v_max[p] = np.full(list_shape, np.nan)
    del temp_log, temp_linear
    #
    if (method_abbr == 'PL') and galactic_integrated:
        logSigmads, alphas, gammas, Umins = \
            np.meshgrid(v_prop['dust.surface.density'][1],
                        v_prop['alpha'][1],
                        10**v_prop['gamma'][1],
                        10**v_prop['logUmin'][1])
        logUbars = np.log10((1 - gammas) * Umins + gammas * Umins *
                            np.log(10**3 / Umins) / (1 - Umins / 10**3))
        idx = np.argsort(logUbars.flatten())
        logUbars_sorted = logUbars.flatten()[idx]
        logUbar_map = np.full(list_shape, np.nan)
        logUbar_min = np.full(list_shape, np.nan)
        logUbar_max = np.full(list_shape, np.nan)
        del logSigmads, alphas, gammas, Umins, logUbars
    #
    if save_pdfs:
        v_pdf = {}
        for p in parameters[method_abbr]:
            v_pdf[p] = np.full([len(v_prop[p][1])] + list_shape, np.nan)
    if rand_cube:
        v_real = {}
        for p in parameters[method_abbr]:
            v_real[p] = np.full([randcubesize] + list_shape, np.nan)
    #
    # Pre fitting variable definitions
    #
    if Voronoi:
        pass  # haha...
    else:
        steps = np.arange(diskmask.size)[diskmask.flatten() == 1]
    total_steps = len(steps)

    def mp_fitting(mpid, mp_var, mp_pdf, mp_rec_sed, mp_chi2, mp_realcube,
                   mp_logUbar):
        qi = int(total_steps * mpid / nop)
        qf = int(total_steps * (mpid + 1) / nop)
        progress = 0.0
        for q in range(qi, qf):
            if (q - qi) / (qf - qi) > progress:
                print(' --mpid', mpid, 'at', str(int(progress * 100)) +
                      '% (' + time.ctime() + ')')
                progress += 0.1
            k = steps[q]
            if Voronoi:
                pass  # haha...
            else:
                i, j = np.unravel_index(k, list_shape)
            #
            """ Calculate covariance matrix """
            #
            sed_vec = sed[i, j].reshape(1, nwl)
            calcov = sed_vec.T * cali_mat2 * sed_vec
            cov_n1 = np.linalg.inv(bkgcov + calcov)
            #
            """ Calculate chi^2 values """
            #
            if import_beta:
                diff = modelss[beta_in[i, j]] - sed[i, j]
            else:
                diff = models - sed[i, j]
            shape0 = list(diff.shape)[:-1]
            shape1 = shape0 + [1, nwl]
            shape2 = shape0 + [nwl, 1]
            chi2 = np.matmul(np.matmul(diff.reshape(shape1), cov_n1),
                             diff.reshape(shape2)).reshape(shape0)
            if np.any(chi2 < 0):
                mask = chi2 < 0
                print(mpid, 'chi2 < 0!!!\n',
                      mpid, 'chi2 =', chi2[mask][0], '\n',
                      mpid, 'diff =', diff[mask][0], '\n',
                      mpid, 'model =', models[mask][0], '\n',
                      mpid, 'SED =', sed[i, j], '\n',
                      mpid, 'cov_n1 =', cov_n1)
            pr = np.exp(-0.5 * chi2)
            #
            """ Save fitting results """
            #
            _vars = []
            _pdfs = []
            for p in parameters[method_abbr]:
                temp_pdf = normalize_pdf(pr, axis_id[p])
                _vars.append(best_fit_and_error(v_prop[p][1], temp_pdf,
                                                islog=v_prop[p][0],
                                                minmax=True))
                if save_pdfs:
                    _pdfs.append(temp_pdf)
            mp_var[q] = _vars
            """ Save Ubar """
            if (method_abbr == 'PL') and galactic_integrated:
                pr_sorted = pr.flatten()[idx] / np.sum(pr)
                # exp
                pexp = np.sum(logUbars_sorted * pr_sorted)
                # 1684
                csp = np.cumsum(pr_sorted)
                csp = csp / csp[-1]
                p16, p84 = np.interp([0.16, 0.84], csp, logUbars_sorted)
                _logubar = [pexp, p16, p84]
                mp_logUbar[q] = _logubar
                del pr_sorted
            """ Save PDFs """
            if save_pdfs:
                mp_pdf[q] = _pdfs
            """ Save randomly chosen randcubesize points from cube """
            if rand_cube:
                # realization pool
                rs = np.random.choice(np.arange(chi2.size), randcubesize,
                                      replace=True,
                                      p=pr.flatten() / np.sum(pr))
                npara = len(parameters[method_abbr])
                _randcube = np.full([npara, randcubesize], np.nan)
                for ri in range(randcubesize):
                    coor = np.unravel_index(rs[ri], chi2.shape)
                    for pi in range(npara):
                        p = parameters[method_abbr][pi]
                        _randcube[pi, ri] = v_prop[p][1][coor[axis_id[p]]]
                """ Save the cube """
                mp_realcube[q] = _randcube
            #
            """ Recover SED from best fit. Save SED and chi^2 values """
            #
            if import_beta:
                if method_abbr == 'SE':
                    rec_sed = SEMBB(wl, _vars[0][0], _vars[1][0], _vars[2][0],
                                    kappa160=kappa160s[beta_in[i, j]])
                elif method_abbr == 'FB':
                    rec_sed = SEMBB(wl, _vars[0][0], _vars[1][0],
                                    beta_in[i, j],
                                    kappa160=kappa160s[beta_in[i, j]])
                elif method_abbr == 'BE':
                    rec_sed = BEMBB(wl, _vars[0][0], _vars[1][0],
                                    beta_in[i, j], lambdac_f, _vars[2][0],
                                    kappa160=kappa160s[beta_in[i, j]])
                elif method_abbr == 'WD':
                    rec_sed = WD(wl, _vars[0][0], _vars[1][0], beta_in[i, j],
                                 _vars[2][0],
                                 kappa160=kappa160s[beta_in[i, j]])
                elif method_abbr == 'PL':
                    rec_sed = PowerLaw(wl, _vars[0][0], _vars[1][0],
                                       _vars[2][0], _vars[3][0],
                                       beta=beta_in[i, j],
                                       kappa160=kappa160s[beta_in[i, j]])
            else:
                if method_abbr == 'SE':
                    rec_sed = SEMBB(wl, _vars[0][0], _vars[1][0], _vars[2][0],
                                    kappa160=kappa160)
                elif method_abbr == 'FB':
                    rec_sed = SEMBB(wl, _vars[0][0], _vars[1][0], beta_f,
                                    kappa160=kappa160)
                elif method_abbr == 'BE':
                    rec_sed = BEMBB(wl, _vars[0][0], _vars[1][0], beta_f,
                                    lambdac_f, _vars[2][0], kappa160=kappa160)
                elif method_abbr == 'WD':
                    rec_sed = WD(wl, _vars[0][0], _vars[1][0], beta_f,
                                 _vars[2][0], kappa160=kappa160)
                elif method_abbr == 'PL':
                    rec_sed = PowerLaw(wl, _vars[0][0], _vars[1][0],
                                       _vars[2][0], _vars[3][0], beta=beta_f,
                                       kappa160=kappa160)
            mp_rec_sed[q] = rec_sed
            diff = rec_sed - sed[i, j]
            shape1 = [1, nwl]
            shape2 = [nwl, 1]
            chi2 = np.matmul(np.matmul(diff.reshape(shape1), cov_n1),
                             diff.reshape(shape2)).reshape(1)
            mp_chi2[q] = chi2

    print("Start fitting", longname, "dust surface density... (" +
          time.ctime() + ')')
    print("Total steps:", total_steps)
    print("Total number of cores:", nop)

    mp_chi2 = mp.Manager().list([0.] * total_steps)
    mp_logUbar = mp.Manager().list([0., 0., 0.] * total_steps)
    mp_rec_sed = mp.Manager().list([[0., 0., 0., 0., 0.]] * total_steps)
    mp_var = mp.Manager().list([[0., 0., 0.]] * total_steps)
    mp_pdf = mp.Manager().list([[0., 0., 0.]] * total_steps) \
        if save_pdfs else -1
    mp_realcube = mp.Manager().list([[0., 0., 0.]] * total_steps) \
        if rand_cube else -1
    processes = [mp.Process(target=mp_fitting,
                            args=(mpid, mp_var, mp_pdf, mp_rec_sed, mp_chi2,
                                  mp_realcube, mp_logUbar))
                 for mpid in range(nop)]
    for p in processes:
        p.start()
    for p in processes:
        p.join()
    del processes
    print("Fitting finished. Start loading results from cores.")
    progress = 0
    for q in range(total_steps):
        if (q + 1) / total_steps > progress:
            print(' --', str(int(progress * 100)) +
                  '% (' + time.ctime() + ')')
            progress += 0.1
        k = steps[q]
        i, j = np.unravel_index(k, diskmask.shape)
        rid = 0
        for p in parameters[method_abbr]:
            v_map[p][0, i, j] = mp_var[q][rid][0]
            v_min[p][i, j] = mp_var[q][rid][1]
            v_max[p][i, j] = mp_var[q][rid][2]
            rid += 1
        #
        """ Save Ubar """
        if (method_abbr == 'PL') and galactic_integrated:
            logUbar_map[i, j] = mp_logUbar[q][0]
            logUbar_min[i, j] = mp_logUbar[q][1]
            logUbar_max[i, j] = mp_logUbar[q][2]
        #
        """ Save PDFs """
        #
        rid = 0
        if save_pdfs:
            for p in parameters[method_abbr]:
                v_pdf[p][:, i, j] = mp_pdf[q][rid]
                rid += 1
        #
        """ Save the randomly selected cube """
        #
        rid = 0
        if rand_cube:
            for p in parameters[method_abbr]:
                v_real[p][:, i, j] = mp_realcube[q][rid]
                rid += 1
        #
        """ Recover SED from best fit. Save SED and chi^2 values """
        #
        recovered_sed[:, i, j] = mp_rec_sed[q]
        chi2_map[i, j] = mp_chi2[q]

    # build error map
    for p in parameters[method_abbr]:
        if v_prop[p][0]:  # log case
            v_map[p][1] = np.max([np.log10(v_max[p]) - np.log10(v_map[p][0]),
                                  np.log10(v_map[p][0]) - np.log10(v_min[p])],
                                 axis=0)
            v_map[p][2] = np.max([v_max[p] - v_map[p][0],
                                  v_map[p][0] - v_min[p]], axis=0)
        else:  # linear case
            v_map[p][1] = np.max([v_max[p] - v_map[p][0],
                                  v_map[p][0] - v_min[p]], axis=0)

    print("Loading finished.")
    #
    outputpath = 'Projects/' + project_name + '/'
    if not os.path.isdir(outputpath):
        os.mkdir(outputpath)
    outputpath += name + '/'
    if not os.path.isdir(outputpath):
        os.mkdir(outputpath)
    if subdir is not None:
        outputpath += subdir + '/'
        if not os.path.isdir(outputpath):
            os.mkdir(outputpath)
    #
    fnhead = outputpath + name + '_'
    fnend = '_' + method_abbr + '.beta=' + betastr + '_' + notes + '.fits'
    os.system('find ' + outputpath + ' -name "*.gz" -delete')
    os.system('find ' + outputpath + ' -name "*.png" -delete')
    #
    if galactic_integrated:
        df = pd.DataFrame()
        for bi in range(nwl):
            df[bands[bi]] = [sed[0, 0, bi]]
        for p in parameters[method_abbr]:
            df[p] = [v_map[p][0, 0, 0]]
            df[p + '.err'] = [v_map[p][1, 0, 0]]
            df[p + '.max'] = [v_max[p][0, 0]]
            df[p + '.min'] = [v_min[p][0, 0]]
        if method_abbr == 'PL':
            df['Tbar'] = 18 * 10**(logUbar_map[0, 0] / (4 + beta_f))
            df['Tbar.max'] = 18 * 10**(logUbar_max[0, 0] / (4 + beta_f))
            df['Tbar.min'] = 18 * 10**(logUbar_min[0, 0] / (4 + beta_f))
        fn = fnhead + 'integrated.csv'
        df.to_csv(fn, index=False)
        #
        for p in parameters[method_abbr]:
            if save_pdfs:
                hdr2 = hdr.copy()
                hdr2['NAXIS'] = 3
                hdr2['NAXIS3'] = v_pdf[p].shape[0]
                hdr2['BUNIT'] = ''
                t1, t2, t3 = round(grid_para[p][0], 1), \
                    round(grid_para[p][1], 1), round(grid_para[p][2], 3)
                hdr2['PDF_MIN'] = t1
                hdr2['PDF_MAX'] = t2
                hdr2.comments['PDF_MAX'] = 'EXCLUSIVE'
                hdr2['PDF_STEP'] = t3
                if v_prop[p][0]:  # log case
                    hdr2.comments['PDF_MIN'] = 'IN LOG.'
                    hdr2.comments['PDF_MAX'] = 'IN LOG. EXCLUSIVE.'
                    hdr2.comments['PDF_STEP'] = 'IN LOG. EXCLUSIVE.'
                    hdr2['PDFARRAY'] = '10**np.arange(' + str(t1) + ', ' + \
                        str(t2) + ', ' + str(t3) + ')'
                else:  # linear case
                    hdr2.comments['PDF_MIN'] = 'LINEAR.'
                    hdr2.comments['PDF_MAX'] = 'LINEAR. EXCLUSIVE.'
                    hdr2.comments['PDF_STEP'] = 'LINEAR. EXCLUSIVE.'
                    hdr2['PDFARRAY'] = 'np.arange(' + str(t1) + ', ' + \
                        str(t2) + ', ' + str(t3) + ')'
                fn = fnhead + p + '.pdf' + fnend
                save_fits_gz(fn, v_pdf[p], hdr2)
            #
            if rand_cube:
                hdr2 = hdr.copy()
                hdr2['NAXIS'] = 3
                hdr2['NAXIS3'] = randcubesize
                hdr2['BUNIT'] = v_prop[p][2][0]
                fn = fnhead + p + '.rlcube' + fnend
                if v_prop[p][0]:  # log case
                    with np.errstate(invalid='ignore'):
                        save_fits_gz(fn, 10**v_real[p], hdr2)
                else:  # linear case
                    save_fits_gz(fn, v_real[p], hdr2)
    else:
        for p in parameters[method_abbr]:
            hdr2 = hdr.copy()
            hdr2['NAXIS'] = 3
            hdr2['NAXIS3'] = 3 if v_prop[p][0] else 2
            hdr2['PLANE0'] = 'Expectation value (linear)'
            if v_prop[p][0]:  # log case
                hdr2['PLANE1'] = 'Error map (dex)'
                hdr2['PLANE2'] = 'Error map (linear)'
            else:  # linear case
                hdr2['PLANE1'] = 'Error map'
            hdr2['BUNIT'] = v_prop[p][2][0]
            fn = fnhead + p + fnend
            save_fits_gz(fn, v_map[p], hdr2)
            #
            hdr2 = hdr.copy()
            hdr2['NAXIS'] = 2
            hdr2['PLANE0'] = 'Maximum possible value (linear)'
            hdr2['BUNIT'] = v_prop[p][2][0]
            fn = fnhead + p + '.max' + fnend
            save_fits_gz(fn, v_max[p], hdr2)
            hdr2['PLANE0'] = 'Minimum possible value (linear)'
            fn = fnhead + p + '.min' + fnend
            save_fits_gz(fn, v_min[p], hdr2)
            #
            if save_pdfs:
                hdr2 = hdr.copy()
                hdr2['NAXIS'] = 3
                hdr2['NAXIS3'] = v_pdf[p].shape[0]
                hdr2['BUNIT'] = ''
                t1, t2, t3 = round(grid_para[p][0], 1), \
                    round(grid_para[p][1], 1), round(grid_para[p][2], 3)
                hdr2['PDF_MIN'] = t1
                hdr2['PDF_MAX'] = t2
                hdr2.comments['PDF_MAX'] = 'EXCLUSIVE'
                hdr2['PDF_STEP'] = t3
                if v_prop[p][0]:  # log case
                    hdr2.comments['PDF_MIN'] = 'IN LOG.'
                    hdr2.comments['PDF_MAX'] = 'IN LOG. EXCLUSIVE.'
                    hdr2.comments['PDF_STEP'] = 'IN LOG. EXCLUSIVE.'
                    hdr2['PDFARRAY'] = '10**np.arange(' + str(t1) + ', ' + \
                        str(t2) + ', ' + str(t3) + ')'
                else:  # linear case
                    hdr2.comments['PDF_MIN'] = 'LINEAR.'
                    hdr2.comments['PDF_MAX'] = 'LINEAR. EXCLUSIVE.'
                    hdr2.comments['PDF_STEP'] = 'LINEAR. EXCLUSIVE.'
                    hdr2['PDFARRAY'] = 'np.arange(' + str(t1) + ', ' + \
                        str(t2) + ', ' + str(t3) + ')'
                fn = fnhead + p + '.pdf' + fnend
                save_fits_gz(fn, v_pdf[p], hdr2)
            #
            if rand_cube:
                hdr2 = hdr.copy()
                hdr2['NAXIS'] = 3
                hdr2['NAXIS3'] = randcubesize
                hdr2['BUNIT'] = v_prop[p][2][0]
                fn = fnhead + p + '.rlcube' + fnend
                if v_prop[p][0]:  # log case
                    with np.errstate(invalid='ignore'):
                        save_fits_gz(fn, 10**v_real[p], hdr2)
                else:  # linear case
                    save_fits_gz(fn, v_real[p], hdr2)
    #
    hdr2 = hdr.copy()
    hdr2['NAXIS'] = 2
    hdr2['PLANE0'] = 'chi-2 map'
    hdr2['BUNIT'] = ''
    fn = fnhead + 'chi2' + fnend
    save_fits_gz(fn, chi2_map, hdr2)
    #
    hdr2 = hdr.copy()
    hdr2['NAXIS'] = 2
    hdr2['PLANE0'] = 'bkg mask'
    hdr2['BUNIT'] = ''
    fn = fnhead + 'bkgmask' + fnend
    save_fits_gz(fn, bkgmask.astype(int), hdr2)
    #
    print(longname, "Datasets saved.")


def models_creation(method_abbr, beta_f, lambdac_f, instr, kappa160, nop):
    if instr == 'spire':
        bands = ['spire250', 'spire350', 'spire500']
    elif instr == 'pacs':
        bands = ['pacs100', 'pacs160']
    elif instr == 'mips':
        bands = ['mips160']
    betas, lambdacs = beta_f, lambdac_f
    Tds, beta2s, wdfracs, alphas, loggammas, logUmins = 0, 0, 0, 0, 0, 0
    if method_abbr == 'PL':
        logSigmads, alphas, loggammas, logUmins = \
            np.meshgrid(v_prop['dust.surface.density'][1],
                        v_prop['alpha'][1],
                        v_prop['gamma'][1],
                        v_prop['logUmin'][1])

        def fitting_model(wl):
            return PowerLaw(wl, 10**logSigmads, alphas, 10**loggammas,
                            logUmins, beta=betas, kappa160=kappa160)
    elif method_abbr == 'SE':
        logSigmads, Tds, betas = \
            np.meshgrid(v_prop['dust.surface.density'][1],
                        v_prop['dust.temperature'][1],
                        v_prop['beta'][1])

        def fitting_model(wl):
            return SEMBB(wl, 10**logSigmads, Tds, betas,
                         kappa160=kappa160)
    elif method_abbr == 'FB':
        logSigmads, Tds = \
            np.meshgrid(v_prop['dust.surface.density'][1],
                        v_prop['dust.temperature'][1])

        def fitting_model(wl):
            return SEMBB(wl, 10**logSigmads, Tds, betas,
                         kappa160=kappa160)
    elif method_abbr == 'BE':
        logSigmads, Tds, beta2s = \
            np.meshgrid(v_prop['dust.surface.density'][1],
                        v_prop['dust.temperature'][1],
                        v_prop['beta2'][1])

        def fitting_model(wl):
            return BEMBB(wl, 10**logSigmads, Tds, betas, lambdacs, beta2s,
                         kappa160=kappa160)
    elif method_abbr == 'WD':
        logSigmads, Tds, wdfracs = \
            np.meshgrid(v_prop['dust.surface.density'][1],
                        v_prop['dust.temperature'][1],
                        v_prop['warm.dust.fraction'][1])

        def fitting_model(wl):
            return WD(wl, 10**logSigmads, Tds, betas, wdfracs,
                      kappa160=kappa160)

    print(" --Constructing", instr, "RSRF model... (" + time.ctime() + ")")
    _rsrf = pd.read_csv("data/RSRF/" + instr + "_rsrf.csv")
    _wl = _rsrf['wavelength'].values

    def mp_models_creation(mpid, mp_model, _wl):
        qi = int(len(_wl) * mpid / nop)
        qf = int(len(_wl) * (mpid + 1) / nop)
        progress = 0.0
        for q in range(qi, qf):
            if (q - qi) / (qf - qi) > progress:
                print(' --mpid', mpid, 'at', str(int(progress * 100)) +
                      '% (' + time.ctime() + ')')
                progress += 0.1
            w = _wl[q]
            mp_model[q] = fitting_model(w)

    mp_model = mp.Manager().list([0.] * len(_wl))
    processes = [mp.Process(target=mp_models_creation,
                            args=(mpid, mp_model, _wl)) for mpid in range(nop)]
    for p in processes:
        p.start()
    for p in processes:
        p.join()
    del processes
    #
    print("Fitting finished. Start loading results from cores.")
    h_models = np.zeros(list(logSigmads.shape) + [len(_wl)])
    for q in range(len(_wl)):
        progress = 0.0
        if (q + 1) / len(_wl) > progress:
            print(' --', str(int(progress * 100)) +
                  '% (' + time.ctime() + ')')
            progress += 0.1
        h_models[..., q] = mp_model[q]
    del mp_model
    #
    for b in bands:
        print("Calculating", b, "RSRF.")
        rsps = _rsrf[b].values
        models = \
            np.sum(h_models * rsps, axis=-1) / \
            np.sum(rsps * _wl / band_wl[b])
        fn = 'models/' + b + '_' + method_abbr + '.beta=' + \
            str(round(beta_f, 2)) + '.fits'
        save_fits_gz(fn, models, None)
        print("Models saved.")


def kappa_calibration(method_abbr, beta_f, lambdac_f=300.0,
                      nop=6, quiet=True):
    assert False  # Haven't corrected the variables yet
    MWSED = np.array([0.71, 1.53, 1.08, 0.56, 0.25]) * 0.97
    bands_small = np.array(['pacs100', 'pacs160', 'spire250', 'spire350',
                            'spire500'])
    nwl = len(bands_small)
    wl = np.array(band_wl[b] for b in bands_small)
    bands = np.array(band_cap[b] for b in bands_small)
    # Correct mode should be 100-Sum_square with Fixen values
    print('################################################')
    print('    Calibrating ' + method_abbr + ' (' + time.ctime() + ')')
    print('################################################')
    logSigmad_step = 0.025
    logSigmad_min = -4.
    logSigmad_max = 1.
    Td_step = 0.5
    Td_min = 5.
    Td_max = 50.
    beta_step = 0.1
    beta_min = -1.0
    beta_max = 4.0
    beta2_step = 0.25
    beta2_min = -1.0
    beta2_max = 4.0
    wdfrac_step = 0.002
    wdfrac_min = 0.0
    wdfrac_max = 0.05
    alpha_step = 0.1  # Remember to avoid alpha==1
    alpha_min = 1.1
    alpha_max = 3.0
    loggamma_step = 0.2
    loggamma_min = -4
    loggamma_max = 0
    logUmin_step = 0.1
    logUmin_min = -2.
    logUmin_max = 1.5
    # Due to memory limit
    if method_abbr == 'PL':
        logSigmad_min = -2.
        logSigmad_max = 0.
        logUmin_min = -1.
        logUmin_max = 0.
        loggamma_max = -1.
    #
    # MW measurement dataset
    #
    DCOU = 10.0 / 100.0
    DUNU = 1.0 / 100.0
    FCOU = 2.0 / 100.0
    FUNU = 0.5 / 100.0
    MWcali_mat2 = np.array([[DUNU + DCOU, 0, 0, 0, 0],
                            [0, FCOU + FUNU, FCOU, FCOU, FCOU],
                            [0, FCOU, FCOU + FUNU, FCOU, FCOU],
                            [0, FCOU, FCOU, FCOU + FUNU, FCOU],
                            [0, FCOU, FCOU, FCOU, FCOU + FUNU]])**2
    MWSigmaD = (1e20 * 1.0079 * u.g / N_A.value).to(u.M_sun).value * \
        ((1 * u.pc).to(u.cm).value)**2 / 150.
    #
    # Build fitting grid
    #
    for iter_ in range(2):
        logSigmads_1d = np.arange(logSigmad_min, logSigmad_max, logSigmad_step)
        betas = beta_f
        if method_abbr == 'PL':
            alphas_1d = np.arange(alpha_min, alpha_max, alpha_step)
            logUmins_1d = np.arange(logUmin_min, logUmin_max, logUmin_step)
            loggammas_1d = np.arange(loggamma_min, loggamma_max, loggamma_step)
            logSigmads, alphas, loggammas, logUmins = \
                np.meshgrid(logSigmads_1d, alphas_1d, loggammas_1d,
                            logUmins_1d)
        else:
            Tds_1d = np.arange(Td_min, Td_max, Td_step)
        if method_abbr == 'SE':
            betas_1d = np.arange(beta_min, beta_max, beta_step)
            logSigmads, Tds, betas = np.meshgrid(logSigmads_1d, Tds_1d,
                                                 betas_1d)
        elif method_abbr == 'FB':
            logSigmads, Tds = np.meshgrid(logSigmads_1d, Tds_1d)
        elif method_abbr == 'BE':
            beta2s_1d = np.arange(beta2_min, beta2_max, beta2_step)
            logSigmads, Tds, beta2s = \
                np.meshgrid(logSigmads_1d, Tds_1d, beta2s_1d)
            lambdacs = np.full(Tds.shape, lambdac_f)
        elif method_abbr == 'WD':
            wdfracs_1d = np.arange(wdfrac_min, wdfrac_max, wdfrac_step)
            logSigmads, Tds, wdfracs = np.meshgrid(logSigmads_1d, Tds_1d,
                                                   wdfracs_1d)
        sigmas = 10**logSigmads
        #
        # Build models
        #
        models = np.zeros(list(logSigmads.shape) + [nwl])
        # Applying RSRFs to generate fake-observed models
        if method_abbr in ['BEMFB', 'BE']:
            def fitting_model(wl):
                return BEMBB(wl, sigmas, Tds, betas, lambdacs, beta2s,
                             kappa160=1.)
        elif method_abbr in ['WD']:
            def fitting_model(wl):
                return WD(wl, sigmas, Tds, betas, wdfracs,
                          kappa160=1.)
        elif method_abbr in ['PL']:
            def fitting_model(wl):
                return PowerLaw(wl, sigmas, alphas, 10**loggammas,
                                logUmins, beta=betas, kappa160=1.)
        else:
            def fitting_model(wl):
                return SEMBB(wl, sigmas, Tds, betas,
                             kappa160=1.)

        def split_herschel(ri, r_, rounds, _wl, wlr, output):
            tic = time.clock()
            rw = ri + r_ * nop
            lenwls = wlr[rw + 1] - wlr[rw]
            last_time = time.clock()
            result = np.zeros(list(logSigmads.shape) + [lenwls])
            if not quiet:
                print("   --process", ri, "starts... (" + time.ctime() +
                      ") (round", (r_ + 1), "of", str(rounds) + ")")
            for i in range(lenwls):
                result[..., i] = fitting_model(_wl[i + wlr[rw]])
                current_time = time.clock()
                # print progress every 10 mins
                if (current_time > last_time + 600.) and (not quiet):
                    last_time = current_time
                    print("     --process", ri,
                          str(round(100. * (i + 1) / lenwls, 1)) +
                          "% Done. (round", (r_ + 1), "of", str(rounds) + ")")
            output.put((ri, rw, result))
            if not quiet:
                print("   --process", ri, "Done. Elapsed time:",
                      round(time.clock()-tic, 3), "s. (" + time.ctime() + ")")

        models = np.zeros(list(logSigmads.shape) + [nwl])
        timeout = 1e-6
        # Applying RSRFs to generate fake-observed models
        instrs = ['PACS', 'SPIRE']
        parallel_rounds = {'SE': 3, 'FB': 1, 'BE': 3, 'WD': 3, 'PL': 12}
        rounds = parallel_rounds[method_abbr]
        for instr in range(2):
            if not quiet:
                print(" --Constructing", instrs[instr], "RSRF model... (" +
                      time.ctime() + ")")
            ttic = time.clock()
            _rsrf = pd.read_csv("data/RSRF/" + instrs[instr] + "_RSRF.csv")
            _wl = _rsrf['Wavelength'].values
            h_models = np.zeros(list(logSigmads.shape) + [len(_wl)])
            wlr = [int(ri * len(_wl) / float(nop * rounds)) for ri in
                   range(nop * rounds + 1)]
            if instr == 0:
                rsps = [_rsrf['PACS_100'].values,
                        _rsrf['PACS_160'].values]
                range_ = range(0, 2)
            elif instr == 1:
                rsps = [[], [], _rsrf['SPIRE_250'].values,
                        _rsrf['SPIRE_350'].values,
                        _rsrf['SPIRE_500'].values]
                range_ = range(2, 5)
            del _rsrf
            # Parallel code
            for r_ in range(rounds):
                if not quiet:
                    print("\n   --" + method_abbr, instrs[instr] + ":Round",
                          (r_ + 1), "of", rounds, '\n')
                q = mp.Queue()
                processes = [mp.Process(target=split_herschel,
                                        args=(ri, r_, rounds, _wl, wlr, q))
                             for ri in range(nop)]
                for p in processes:
                    p.start()
                for p in processes:
                    p.join(timeout)
                for p in processes:
                    ri, rw, result = q.get()
                    if not quiet:
                        print("     --Got result from process", ri)
                    h_models[..., wlr[rw]:wlr[rw+1]] = result
                    del ri, rw, result
                del processes, q, p
            # Parallel code ends
            if not quiet:
                print("   --Calculating response function integrals")
            for i in range_:
                models[..., i] = \
                    np.sum(h_models * rsps[i], axis=-1) / \
                    np.sum(rsps[i] * _wl / wl[i])
            del _wl, rsps, h_models, range_
            if not quiet:
                print("   --Done. Elapsed time:", round(time.clock()-ttic, 3),
                      "s.\n")
        #
        # Start fitting
        #
        tic = time.clock()
        temp_matrix = np.empty_like(models)
        diff = models - MWSED
        sed_vec = MWSED.reshape(1, 5)
        yerr = MWSED * np.sqrt(np.diagonal(MWcali_mat2))
        cov_n1 = np.linalg.inv(sed_vec.T * MWcali_mat2 * sed_vec)
        for j in range(nwl):
            temp_matrix[..., j] = np.sum(diff * cov_n1[:, j], axis=-1)
        chi2 = np.sum(temp_matrix * diff, axis=-1)
        r_chi2 = chi2 / (nwl - ndims[method_abbr])
        """ Find the (s, t) that gives Maximum likelihood """
        am_idx = np.unravel_index(chi2.argmin(), chi2.shape)
        """ Probability and mask """
        mask = r_chi2 <= np.nanmin(r_chi2) + 50.
        pr = np.exp(-0.5 * chi2)
        print('\nIteration', str(iter_ + 1))
        print('Best fit r_chi^2:', r_chi2[am_idx])
        """ kappa 160 """
        logkappa160s = logSigmads - np.log10(MWSigmaD)
        logkappa160, logkappa160_err, _1, _2, _3, _4 = \
            best_fit_and_error(logkappa160s, pr, 'logkappa_160')
        kappa160 = 10**logkappa160
        logSigmad, _, _1, _2, _3, _4 = \
            best_fit_and_error(logSigmads, pr, 'logSigmads')
        #
        logSigmad_min = logSigmad - 0.2
        logSigmad_max = logSigmad + 0.2
        # All steps
        logSigmad_step = 0.002
        Td_step = 0.1
        beta_step = 0.02
        beta2_step = 0.02
        wdfrac_step = 0.0005
        alpha_step = 0.01  # Remember to avoid alpha==1
        loggamma_step = 0.1
        logUmin_step = 0.01
        print('Best fit kappa160:', kappa160)
        wl_complete = np.linspace(1, 1000, 1000)
        #
        fn = 'hdf5_MBBDust/Calibration_' + str(round(beta_f, 2)) + '.h5'
        hf = h5py.File(fn, 'a')
        try:
            del hf[method_abbr]
        except KeyError:
            pass
        grp = hf.create_group(method_abbr)
        grp['kappa160'] = kappa160
        grp['logkappa160'], grp['logkappa160_err'] = \
            logkappa160, logkappa160_err
        if method_abbr == 'SE':
            samples = np.array([logkappa160s[mask], Tds[mask], betas[mask],
                                r_chi2[mask]])
            labels = [r'$\log\kappa_{160}$', r'$T$', r'$\beta$',
                      r'$\tilde{\chi}^2$']
            T, T_err, _1, _2, _3, _4 = \
                best_fit_and_error(Tds, pr, 'T')
            beta, beta_err, _1, _2, _3, _4 = \
                best_fit_and_error(betas, pr, 'beta')
            Td_min = T - 1.5
            Td_max = T + 1.5
            beta_min = beta - 0.3
            beta_max = beta + 0.3
            grp['T'], grp['T_err'] = T, T_err
            grp['beta'], grp['beta_err'] = beta, beta_err
            mode_integrated = \
                z0mg_RSRF(wl_complete, SEMBB(wl_complete, MWSigmaD, T, beta,
                                             kappa160=kappa160), bands)
            model_complete = SEMBB(wl_complete, MWSigmaD, T, beta,
                                   kappa160=kappa160)
            gordon_integrated = \
                z0mg_RSRF(wl_complete, SEMBB(wl_complete, MWSigmaD, 17.2, 1.96,
                                             9.6 * np.pi), bands)
            model_gordon = SEMBB(wl_complete, MWSigmaD, 17.2,
                                 1.96, 9.6 * np.pi)
        elif method_abbr == 'FB':
            samples = np.array([logkappa160s[mask], Tds[mask], r_chi2[mask]])
            labels = [r'$\log\kappa_{160}$', r'$T$', r'$\tilde{\chi}^2$']
            T, T_err, _1, _2, _3, _4 = \
                best_fit_and_error(Tds, pr, 'T')
            Td_min = T - 1.5
            Td_max = T + 1.5
            grp['T'], grp['T_err'] = T, T_err
            mode_integrated = \
                z0mg_RSRF(wl_complete, SEMBB(wl_complete, MWSigmaD, T, beta_f,
                                             kappa160=kappa160), bands)
            model_complete = SEMBB(wl_complete, MWSigmaD, T, beta_f,
                                   kappa160=kappa160)
            gordon_integrated = \
                z0mg_RSRF(wl_complete, SEMBB(wl_complete, MWSigmaD, 17.2, 1.96,
                                             9.6 * np.pi), bands)
            model_gordon = SEMBB(wl_complete, MWSigmaD, 17.2,
                                 1.96, 9.6 * np.pi)
        elif method_abbr == 'BE':
            samples = np.array([logkappa160s[mask], Tds[mask], beta2s[mask],
                                r_chi2[mask]])
            labels = [r'$\log\kappa_{160}$', r'$T$', r'$\beta_2$',
                      r'$\tilde{\chi}^2$']
            T, T_err, _1, _2, _3, _4 = \
                best_fit_and_error(Tds, pr, 'T')
            beta2, beta2_err, _1, _2, _3, _4 = \
                best_fit_and_error(beta2s, pr, 'beta2')
            Td_min = T - 1.5
            Td_max = T + 1.5
            beta2_min = beta2 - 0.3
            beta2_max = beta2 + 0.3
            grp['T'], grp['T_err'] = T, T_err
            grp['beta2'], grp['beta2_err'] = beta2, beta2_err
            mode_integrated = \
                z0mg_RSRF(wl_complete, BEMBB(wl_complete, MWSigmaD, T, beta_f,
                                             lambdac_f, beta2,
                                             kappa160=kappa160), bands)
            model_complete = BEMBB(wl_complete, MWSigmaD, T, beta_f,
                                   lambdac_f, beta2, kappa160=kappa160)
            e500 = 0.48
            gbeta2 = np.log(e500 + 1) / np.log(294. / 500.) + 2.27
            gordon_integrated = \
                z0mg_RSRF(wl_complete, BEMBB(wl_complete, MWSigmaD, 16.8, 2.27,
                                             294, gbeta2,
                                             11.6 * np.pi), bands)
            model_gordon = BEMBB(wl_complete, MWSigmaD, 16.8, 2.27, 294,
                                 gbeta2, 11.6 * np.pi)
        elif method_abbr == 'WD':
            samples = np.array([logkappa160s[mask], Tds[mask], wdfracs[mask],
                                r_chi2[mask]])
            labels = [r'$\log\kappa_{160}$', r'$T$', r'wdfrac',
                      r'$\tilde{\chi}^2$']
            T, T_err, _1, _2, _3, _4 = \
                best_fit_and_error(Tds, pr, 'T')
            wdfrac, wdfrac_err, _1, _2, _3, _4 = \
                best_fit_and_error(wdfracs, pr, 'wdfrac')
            Td_min = T - 1.5
            Td_max = T + 1.5
            wdfrac_min = 0.0
            wdfrac_max = wdfrac + 0.006
            grp['T'], grp['T_err'] = T, T_err
            grp['wdfrac'], grp['wdfrac_err'] = wdfrac, wdfrac_err
            mode_integrated = \
                z0mg_RSRF(wl_complete, WD(wl_complete, MWSigmaD, T, beta_f,
                                          wdfrac, kappa160=kappa160), bands)
            model_complete = WD(wl_complete, MWSigmaD, T, beta_f, wdfrac,
                                kappa160=kappa160)
            e500 = 0.91
            nu500 = (c / 500 / u.um).to(u.Hz).value
            gwdfrac = e500 * B_fast(15., nu500) / B_fast(6., nu500)
            gordon_integrated = \
                z0mg_RSRF(wl_complete, WD(wl_complete, MWSigmaD, 15., 2.9,
                                          gwdfrac, kappa160=517. * np.pi,
                                          WDT=6.), bands)
            model_gordon = WD(wl_complete, MWSigmaD, 15., 2.9, gwdfrac,
                              kappa160=517. * np.pi, WDT=6.)
        elif method_abbr == 'PL':
            samples = np.array([logkappa160s[mask], loggammas[mask],
                                alphas[mask], logUmins[mask], r_chi2[mask]])
            labels = [r'$\log\kappa_{160}$', r'$\log\gamma$', r'$\alpha$',
                      r'\log U_{min}', r'$\tilde{\chi}^2$']
            loggamma, loggamma_err, _1, _2, _3, _4 = \
                best_fit_and_error(loggammas, pr, 'loggamma')
            alpha, alpha_err, _1, _2, _3, _4 = \
                best_fit_and_error(alphas, pr, 'alpha')
            logUmin, logUmin_err, _1, _2, _3, _4 = \
                best_fit_and_error(logUmins, pr, 'logUmin')
            alpha_min = max(alpha - 0.3, 1.1)
            alpha_max = alpha + 0.3
            loggamma_min = loggamma - 0.3
            loggamma_max = min(loggamma + 0.3, 0.)
            logUmin_min = logUmin - 0.1
            logUmin_max = logUmin + 0.1
            grp['loggamma'], grp['loggamma_err'] = loggamma, loggamma_err
            grp['alpha'], grp['alpha_err'] = alpha, alpha_err
            grp['logUmin'], grp['logUmin_err'] = logUmin, logUmin_err
            mode_integrated = \
                z0mg_RSRF(wl_complete, PowerLaw(wl_complete, MWSigmaD, alpha,
                                                10**loggamma, logUmin,
                                                beta=beta_f,
                                                kappa160=kappa160), bands)
            model_complete = PowerLaw(wl_complete, MWSigmaD, alpha,
                                      10**loggamma, logUmin, beta=beta_f,
                                      kappa160=kappa160)
            gordon_integrated = \
                z0mg_RSRF(wl_complete, SEMBB(wl_complete, MWSigmaD, 17.2, 1.96,
                                             9.6 * np.pi), bands)
            model_gordon = SEMBB(wl_complete, MWSigmaD, 17.2, 1.96,
                                 9.6 * np.pi)
        hf.close()
    #
    del samples, labels
    """
    fig = corner(samples.T, labels=labels, quantities=(0.16, 0.84),
                 show_titles=True, title_kwargs={"fontsize": 12})
    with PdfPages('output/_CALI_' + method_abbr + '.pdf') as pp:
        pp.savefig(fig)
    """
    fig, ax = plt.subplots(figsize=(10, 7.5))
    ax.loglog(wl_complete, model_gordon, label='G14EXP')
    ax.loglog(wl, mode_integrated, 'x', ms=15, label='fitting (int)')
    ax.loglog(wl_complete, model_complete, label='fitting')
    ax.errorbar(wl, MWSED, yerr, label='MWSED')
    ax.loglog(wl, gordon_integrated, 'x', ms=15, label='G14 (int)')
    ax.legend()
    ax.set_ylim(0.03, 3.0)
    ax.set_xlim(80, 1000)
    ax.set_xlabel(r'SED [$MJy\,sr^{-1}\,(10^{20}(H\,Atom)\,cm^{-2})^{-1}$]')
    ax.set_ylabel(r'Wavelength ($\mu m$)')
    with PdfPages('output/_CALI_' + method_abbr + '_MODEL.pdf') as pp:
        pp.savefig(fig)
    print(" --Done. Elapsed time:", round(time.clock()-tic, 3), "s.")
