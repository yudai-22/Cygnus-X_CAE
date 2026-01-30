from astropy.io import fits
import numpy as np
from matplotlib import pyplot as plt
import aplpy
from astropy.wcs import WCS

import sys
import glob
sys.path.append("/home/cygnus/fujimoto/Cygnus-X_Molecular_Cloud_Analysis/process_tools")
from cut_resize_tools import convolve_vaxis, picking, gaussian_filter


def v2ch(v, w): # v(km/s)をchに変える
    x_tempo, y_tempo, v_tempo   = w.wcs_pix2world(0, 0, 0, 0)
    x_ch, y_ch, v_ch   = w.wcs_world2pix(x_tempo, y_tempo, v*1000.0, 0)
    v_ch = int(round(float(v_ch), 0))
    return v_ch


def del_header_key(header, keys): # headerのkeyを消す
    import copy
    h = copy.deepcopy(header)
    for k in keys:
        try:
            del h[k]
        except:
            pass
    return h

vsmooth = 5
thresh = 1
sigma = 2
sch_rms = 10
ech_rms = 90


fits_path_all = glob.glob("/home/cygnus/fujimoto/Cygnus-X_Molecular_Cloud_Analysis/fits/Cygnus-X/NRO45m/*.fits")
for fits_path in fits_path_all:
    hdu = fits.open(fits_path)[0]
    raw_d = hdu.data
    header = hdu.header
    
    _data = raw_d.copy()
    vconv = convolve_vaxis(_data, vsmooth)
    
    _mask = vconv.copy()
    rms = np.nanstd(vconv[sch_rms:ech_rms], axis=0)
    _mask[np.where(_mask < rms*sigma)] = 0
    ndata, _ = picking(_mask, raw_d, thresh)   
    
    new_hdu = fits.PrimaryHDU(ndata, header)
    fits_path = fits_path.replace(".fits", "_zeroing.fits")
    new_hdu.writeto(fits_path, overwrite=True)