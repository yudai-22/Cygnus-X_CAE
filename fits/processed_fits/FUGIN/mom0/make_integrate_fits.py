from astropy.io import fits
import numpy as np
from matplotlib import pyplot as plt
import aplpy
from astropy.wcs import WCS

import sys
import glob
# sys.path.append("/home/cygnus/fujimoto/Cygnus-X_Molecular_Cloud_Analysis/process_tools")
# from cut_resize_tools import convolve_vaxis, picking, gaussian_filter


def del_header_key(header, keys): # headerのkeyを消す
    import copy
    h = copy.deepcopy(header)
    for k in keys:
        try:
            del h[k]
        except:
            pass
    return h

def make_new_hdu_integ(hdu, v_start_ch, v_end_ch): # 積分強度のhduを作る
    data = hdu.data
    header = hdu.header
    # start_ch, end_ch = v2ch(v_start_wcs, w), v2ch(v_end_wcs, w)
    new_data = np.nansum(data[v_start_ch:v_end_ch+1], axis=0)*np.abs(header["CDELT3"])/1000.0
    header = del_header_key(header, ["CRVAL3", "CRPIX3", "CRVAL3", "CDELT3", "CUNIT3", "CTYPE3", "CROTA3", "NAXIS3", "PC1_3", "PC2_3", "PC3_3", "PC3_1", "PC3_2"])
    header["NAXIS"] = 2
    new_hdu = fits.PrimaryHDU(new_data, header)
    return new_hdu


fits_path_all = glob.glob("/home/cygnus/fujimoto/Cygnus-X_Molecular_Cloud_Analysis/Bubble_analysis/All_Bubble_Analysis/Analysis/Fits/Zeroing_Fits/12CO/**/*12CO*.fits")
fits_path_all.sort()

for fits_path in fits_path_all:
    hdu = fits.open(fits_path)[0]
    raw_d = hdu.data
    header = hdu.header

    #積分するチャンネルを指定
    v_start_ch = 0
    v_end_ch = len(raw_d)

    #積分の実行
    new_hdu = make_new_hdu_integ(hdu, v_start_ch, v_end_ch)

    #データの保存
    name = fits_path.split("/")[-1]
    fits_path = name.replace(".fits", "_mom0.fits")
    new_hdu.writeto(fits_path, overwrite=True)
    print(f"completed maiking {name} mom0 fits")