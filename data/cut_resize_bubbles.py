import astropy.io.fits as fits
from astropy.wcs import WCS
import pandas as pd

import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from scipy.signal import fftconvolve
import random

import scipy.ndimage 
from astropy.convolution import convolve, convolve_fft, Gaussian2DKernel, Tophat2DKernel, Gaussian1DKernel
from astropy.modeling.models import Gaussian2D
import scipy.signal
from astropy.wcs import WCS

from Astronomy import *
from Make_Data_Tools import *
from cut_resize_tools import *

fits_path = "/home/filament/fujimoto/fits/Cygnus_sp16_vs-40_ve040_dv0.25_12CO_Tmb.fits"
integ_hdu = fits.open("/home/filament/fujimoto/fits/CygnusX_sp16_120-230_integrate_map.fits")[0]

w_high = WCS(fits_path)
hdu = fits.open(fits_path)[0]


# 1 arcsec = 1/3600 degree
arcsec = 1 / 3600  # 度
radian = np.deg2rad(arcsec)  # ラジアンに変換

tan_value = np.tan(radian)

print(f"tan(1 arcsec) = {tan_value}")



vsmooth = 5 #input smoothing kernel size in pix (v direction)~3-5
thresh = 1 #threthould voxel number 200-700
sigma = 1 #threthold value (after smoothing)2-4

sch_rms = 10 #select start ch to calculate the rms (after smoothing)
ech_rms = 90 #select end ch to calculate the rms (after smoothing)

sch_ii=121
ech_ii=241

integrate_layer_num = 30

hdu = fits.open(fits_path)[0]
raw_d = hdu.data
head = hdu.header


cygnus_infer_catalogue = pd.read_csv("/home/filament/fujimoto/Cygnus-X_CAE/data/cygnus_infer_catalogue.csv")
cygnus_infer_catalogue


ra_min = cygnus_infer_catalogue['ra_min']
ra_max = cygnus_infer_catalogue['ra_max']
dec_min = cygnus_infer_catalogue['dec_min']
dec_max = cygnus_infer_catalogue['dec_max']

r_u_icrs_list = list(zip(ra_min, dec_min))
l_u_icrs_list = list(zip(ra_max, dec_min))
r_t_icrs_list =list(zip(ra_min, dec_max))
l_t_icrs_list =list(zip(ra_max, dec_max))


r_u_galactic_list = []
l_u_galactic_list = []
r_t_galactic_list = []
l_t_galactic_list =[]

for ra,dec in r_u_icrs_list:
    GLON, GLAT = icrs_to_galactic(ra, dec)
    galactic = [GLON,GLAT]
    r_u_galactic_list.append(galactic)

for ra,dec in l_u_icrs_list:
    GLON, GLAT = icrs_to_galactic(ra, dec)
    galactic = [GLON,GLAT]
    l_u_galactic_list.append(galactic)

for ra,dec in r_t_icrs_list:
    GLON, GLAT = icrs_to_galactic(ra, dec)
    galactic = [GLON,GLAT]
    r_t_galactic_list.append(galactic)

for ra,dec in l_t_icrs_list:
    GLON, GLAT = icrs_to_galactic(ra, dec)
    galactic = [GLON,GLAT]
    l_t_galactic_list.append(galactic)

babble_region_glactic = list(zip(r_u_galactic_list, l_u_galactic_list, r_t_galactic_list, l_t_galactic_list))


wcs = WCS(integ_hdu.header)

babble_region_pix = []
for i in range(len(babble_region_glactic)):
    babble_region = babble_region_glactic[i]

    region_list = []
    for j in range(len(babble_region)):
        region_pix = wcs.wcs_world2pix([list(babble_region[j])], 1)
        region_list.append(region_pix)

    babble_region_pix.append(region_list)


cutting_data_list = []
data = raw_d.copy()
for bubble_num in range(len(babble_region_pix)):
    x_min = babble_region_pix[bubble_num][3][0][0]
    x_max = babble_region_pix[bubble_num][0][0][0]
    y_min = babble_region_pix[bubble_num][1][0][1]
    y_max = babble_region_pix[bubble_num][2][0][1]
    
    x_center = int((x_min + x_max) // 2)
    y_center = int((y_min + y_max) // 2)
    
    x_range = int(((x_max - x_min)*2) // 2)
    y_range = int(((y_max - y_min)*2) // 2)

    cutting_data = data[121:241, y_center-y_range:y_center+y_range, x_center-x_range:x_center+x_range]
    cutting_data_list.append(cutting_data)

#中身が無いデータを削除
deleted_indices = []
filtered_list = [
    data for idx, data in enumerate(cutting_data_list) 
    if not np.all((data == 0) | np.isnan(data))
    or deleted_indices.append(idx)  # 削除されたインデックスを記録
]

# 削除されたインデックスを表示
print("削除されたデータのインデックス:", deleted_indices)
print("削除された数: ", len(deleted_indices))
print("残ったデータ数: ",len(filtered_list))


ndata_conv_list = []
for mask_num in tqdm(range(len(filtered_list))):
    data = filtered_list[mask_num]
    data_cp = data.copy()
    vconv = convolve_vaxis(data_cp,vsmooth)
    d1 = data_cp.copy()
    
    _mask = d1.copy()
    rms_conv = np.nanstd(d1[sch_rms:ech_rms],axis=0)
    _mask[np.where(_mask<rms_conv*sigma)]=0
    ndata,areas = picking(_mask, data_cp, thresh)
    # ndata_conv = gaussian_filter(ndata.copy())
    
    ndata_conv_list.append(integrate_to_x_layers(ndata, 30))


remove_list = [5, 9, 10, 12, 17, 19, 21, 34]
clear_bubble_list = [data for i, data in enumerate(ndata_conv_list) if i not in remove_list]

print(len(ndata_conv_list))
print(len(remove_list))
print(len(clear_bubble_list))

