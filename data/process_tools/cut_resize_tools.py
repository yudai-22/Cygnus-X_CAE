import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from functools import partial

from scipy.signal import fftconvolve
from scipy import signal
import scipy.ndimage

import astropy.io.fits as fits
from astropy.convolution import convolve, convolve_fft, Gaussian2DKernel, Tophat2DKernel, Gaussian1DKernel
from astropy.modeling.models import Gaussian2D
import scipy.signal

from npy_append_array import NpyAppendArray
import psutil
from tqdm.notebook import tqdm

from Astronomy import *
from Make_Data_Tools import *


# v畳み込み、mask、zeroing、30層積分、したうえで閾値設定
def maximum_value_determination(mode, data, vsmooth, sch_rms, ech_rms, sch_ii, 
                                ech_ii, sigma, thresh, integrate_layer_num, percentile=None):
    """
    modeには
    "percentile", 
    "sigma"
    のどちらかを入れる
    """

    data = data.copy()
    print('======== maximum determination ========')
    print('-------- making v_convolution data --------')
    _data = data.copy()
    vconv = convolve_vaxis(_data, vsmooth)
    # vconv[np.isnan(vconv)] = 0
    # xyconv = gaussian_filter(vconv)

    print('-------- masking data --------')
    _mask = vconv.copy()
    rms = np.nanstd(vconv[sch_rms:ech_rms], axis=0)
    _mask[np.where(_mask < rms*sigma)] = 0
    ndata, _ = picking(_mask, data, thresh)
    ndata = ndata[sch_ii:ech_ii]

    print('-------- integrating and xy_convolution --------')
    ndata = integrate_to_x_layers(ndata, integrate_layer_num)
    ndata = gaussian_filter(ndata)
    
    if mode == "percentile":
        result = np.nanpercentile(ndata, percentile)
    elif mode == "sigma":
        result = np.nanstd(ndata)
    else:
        print("The value entered for mode is incorrect.")
        
    print(f"{mode} value is", "{:.2f}".format(result))
    return result


def process_data_segment(data, vsmooth, sch_rms, ech_rms, sch_ii, ech_ii, sigma, thresh, integrate_layer_num):
    _data = data.copy()
    vconv = convolve_vaxis(_data, vsmooth)

    _mask = vconv.copy()
    rms = np.nanstd(vconv[sch_rms:ech_rms], axis=0)
    _mask[np.where(_mask < rms*sigma)] = 0
    ndata, _ = picking(_mask, data, thresh)
    ndata = ndata[sch_ii:ech_ii]
    ndata = integrate_to_x_layers(ndata, integrate_layer_num)
    return ndata


def convolve_vaxis(data, width_v):
    gauss_kernel_1d = Gaussian1DKernel(width_v/(np.sqrt(2*np.log(2.))*2.))
    nz, ny, nx = data.shape
    _data = data.reshape(nz, ny*nx).T
    _new_data = [scipy.signal.convolve(_d, gauss_kernel_1d,'same') for _d in _data]
    new_data = np.array(_new_data).T.reshape(nz, ny, nx)
    return new_data


def picking(data, org_data, threshold_size):
    import numpy
    import scipy.ndimage
    
    data = data.copy()
    
    nanmask = numpy.isnan(data)
    data[nanmask] = 0
    
    data_op = scipy.ndimage.binary_opening(data)
    data_labels, data_nums = scipy.ndimage.label(data_op)

    data_areas = scipy.ndimage.sum(data_op, data_labels, numpy.arange(data_nums+1))

    small_size_mask = data_areas < threshold_size
    small_mask = small_size_mask[data_labels.ravel()].reshape(data_labels.shape)
    
    data[nanmask] = numpy.nan
    org_data[small_mask] = 0
    return org_data, data_areas

def normalization(data_list):
    norm_list = []
    for i in range(len(data_list)):
        data = data_list[i]
        norm_data = (data - np.min(data)) / (np.max(data) - np.min(data))
        norm_list.append(norm_data)
    return norm_list

def normalization_thresh(data_list, max_thresh):
    norm_list = []
    for i in range(len(data_list)):
        data = data_list[i]
        norm_data = (data - np.min(data)) / (max_thresh - np.min(data))
        norm_list.append(norm_data)
    return norm_list


def normalization_sigma(data_list, sigma, multiply):
    max_thresh = sigma * multiply
    norm_list = []
    
    for i in range(len(data_list)):
        data = data_list[i]
        
        if np.max(data) <= max_thresh:
            norm_data = (data - np.min(data)) / (max_thresh - np.min(data))
        else:
            norm_data = (data - np.min(data)) / (np.max(data) - np.min(data))
            
        norm_list.append(norm_data)
    return norm_list


def parallel_processing(function, target, *args, **kwargs):
    # functionに固定引数を設定
    partial_function = partial(function, *args, **kwargs)

    # 並列処理の実行
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(partial_function, target))
    return results


def select_conv(data, obj_size, obj_sig):
    if data.shape[2] > obj_size:
        # print(data.shape[2])
        fwhm = (data.shape[2] / obj_size) * 2#何ための×2??
        sig3 = fwhm / (2 * (2 * np.log(2)) ** (1 / 2))
        sig2 = (sig3**2 - obj_sig**2) ** (1 / 2)
        
        kernel = np.outer(signal.gaussian(8 * round(sig2) + 1, sig2), signal.gaussian(8 * round(sig2) + 1, sig2))
        kernel1 = kernel / np.sum(kernel)
        
        conv_list = []
        for k in range(data.shape[0]):
            cut_data_k = data[k, :, :]
            lurred_k = signal.fftconvolve(cut_data_k, kernel1, mode="same")
            conv_list.append(lurred_k[None, :, :])
    
        pi = np.concatenate(conv_list, axis=0)
        # print(pi.shape)
        pi = gaussian_filter(pi)
    else:
        pi = gaussian_filter(data)
        
    return pi    


def proccess_npyAppend(file_name, data):
    shape = data[0].shape
    num_arrays = len(data)

    mem = psutil.virtual_memory()
    total = mem.total
    used = mem.used
    free = mem.available
    
    print(f"総メモリ: {total / 2**30:.2f} GB")
    print(f"使用中のメモリ: {used / 2**30:.2f} GB")
    print(f"使用可能なメモリ: {free / 2**30:.2f} GB")
    
    data_byte = np.prod(shape) * 8
    print(f"\n一つあたりのデータ容量: {data_byte/ 2**20:.2f} MB")
    print(f"総データ容量: {(data_byte * num_arrays) / 2**30:.2f} GB")
    
    batch_size = (free // data_byte) // 10
    print(f"\n総データ数: {len(data)}")
    print(f"バッチサイズ:\n {batch_size}")

    file_name = file_name + ".npy"
    saved_file = NpyAppendArray(file_name)
    
    for i in tqdm(range(0, len(data), batch_size)):
      batch_data = np.asarray(data[i:i + batch_size])
      batch_data = np.ascontiguousarray(batch_data)
      saved_file.append(batch_data)

    saved_file.close()
    print("The save has completed!!")