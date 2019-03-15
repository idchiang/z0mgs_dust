import os
import time
import numpy as np
import pandas as pd
import multiprocessing as mp
from astropy.io.fits import Header
from .utils import axis_ids, v_prop, grid_para, band_wl, SEMBB, BEMBB, \
    abspath_and_sep
from ..utils import save_fits_gz


def template_generator(template_name='user_defined',
                       model_name='SE',
                       beta_f=2.0,
                       lambdac_f=300.0,
                       instr='spire',
                       kappa_160=1.0,
                       parallel_mode=False,
                       num_proc=5):
    # Find the path of this script
    abs_path, sep = abspath_and_sep()
    #
    if instr == 'spire':
        bands = ['spire250', 'spire350', 'spire500']
    elif instr == 'pacs':
        bands = ['pacs100', 'pacs160']
    elif instr == 'mips':
        bands = ['mips160']
    betas, lambdacs = beta_f, lambdac_f
    Tds, beta2s = 0, 0
    if model_name == 'SE':
        logSigmads, Tds, betas = \
            np.meshgrid(v_prop['dust.surface.density'][1],
                        v_prop['dust.temperature'][1],
                        v_prop['beta'][1])

        def fitting_model(wl):
            return SEMBB(wl, 10**logSigmads, Tds, betas,
                         kappa_160=kappa_160)
    elif model_name == 'FB':
        logSigmads, Tds = \
            np.meshgrid(v_prop['dust.surface.density'][1],
                        v_prop['dust.temperature'][1])

        def fitting_model(wl):
            return SEMBB(wl, 10**logSigmads, Tds, betas,
                         kappa_160=kappa_160)
    elif model_name == 'BE':
        logSigmads, Tds, beta2s = \
            np.meshgrid(v_prop['dust.surface.density'][1],
                        v_prop['dust.temperature'][1],
                        v_prop['beta2'][1])

        def fitting_model(wl):
            return BEMBB(wl, 10**logSigmads, Tds, betas, lambdacs, beta2s,
                         kappa_160=kappa_160)
    print(" --Constructing", instr, "RSRF model... (" + time.ctime() + ")")
    fn = abs_path + 'RSRF' + sep + instr + "_rsrf.csv"
    _rsrf = pd.read_csv(fn)  # Move rsrf files
    _wl = _rsrf['wavelength'].values
    #
    # Start calculating the complete SED
    if parallel_mode:
        print(' --Parallel mode on. Will use', num_proc, 'processors.')

        def mp_models_creation(mpid, mp_model, _wl):
            qi = int(len(_wl) * mpid / num_proc)
            qf = int(len(_wl) * (mpid + 1) / num_proc)
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
                                args=(mpid, mp_model, _wl))
                     for mpid in range(num_proc)]
        for p in processes:
            p.start()
        for p in processes:
            p.join()
        del processes
        #
        print("Fitting finished. Start loading results from cores.")
        h_models = np.zeros(list(logSigmads.shape) + [len(_wl)])
        progress = 0.0
        for q in range(len(_wl)):
            if (q + 1) / len(_wl) > progress:
                print(' --', str(int(progress * 100)) +
                      '% (' + time.ctime() + ')')
                progress += 0.1
            h_models[..., q] = mp_model[q]
        del mp_model
    else:
        print(' --Parallel mode off.')
        progress = 0.0
        h_models = np.zeros(list(logSigmads.shape) + [len(_wl)])
        for q in range(len(_wl)):
            if (q + 1) / len(_wl) > progress:
                print('Progress:', str(int(progress * 100)) +
                      '% (' + time.ctime() + ')')
                progress += 0.1
            w = _wl[q]
            h_models[..., q] = fitting_model(w)
    #
    # RSRF integral and save
    tpath = abs_path + 'templates' + sep + template_name + sep
    if not os.path.isdir(tpath):
        os.mkdir(tpath)
    #
    hdr = Header()
    ps = list(axis_ids[model_name].keys())
    for i in range(len(ps)):
        for p in ps:
            if axis_ids[model_name][p] == i:
                hdr['P' + str(i) + 'MIN'] = grid_para[p][0]
                hdr['P' + str(i) + 'MAX'] = grid_para[p][1]
                hdr['P' + str(i) + 'STEP'] = grid_para[p][2]
                hdr.comments['P' + str(i) + 'MIN'] = p
    if model_name != 'SE':
        hdr['BETA'] = beta_f
    if model_name == 'BE':
        hdr['LAMBDAC'] = lambdac_f
    #
    for b in bands:
        print("Calculating", b, "RSRF.")
        rsps = _rsrf[b].values
        models = \
            np.sum(h_models * rsps, axis=-1) / \
            np.sum(rsps * _wl / band_wl[b])
        fn = tpath + b + '_' + model_name
        save_fits_gz(fn, models, hdr)
    print("Models saved.")
