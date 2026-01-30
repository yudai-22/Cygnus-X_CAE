import astropy
import numpy as np
from processing import remove_nan, normalize_rp
import copy


def extract_spectral(w_co, fugin_cube_fits, each_catalogue):

    # **************************************
    bubble_x_min, bubble_y_min, _ = w_co.all_world2pix(each_catalogue['ra_max'], each_catalogue['dec_min'], 0, 0)
    bubble_x_max, bubble_y_max, _ = w_co.all_world2pix(each_catalogue['ra_min'], each_catalogue['dec_max'], 0, 0)
    radius_pix = (bubble_y_max - bubble_y_min) * 1/2
    # **************************************

    cut_data = fugin_cube_fits.data[:, 
    int(bubble_y_min - radius_pix):int(bubble_y_max + radius_pix), 
    int(bubble_x_min - radius_pix):int(bubble_x_max + radius_pix)]
    # print(radius_pix)
    # print(int(bubble_y_min - radius_pix), int(bubble_y_max + radius_pix), int(bubble_x_min - radius_pix), int(bubble_x_max + radius_pix))

    return cut_data



def make_spitzer_fits(spitzer_rfits, spitzer_gfits, w_spitzer, data, row):
    header = spitzer_rfits.header
    x, y = w_spitzer.all_world2pix((row["ra_max"]+row["ra_min"])/2, 
                                   (row["dec_min"]+row["dec_max"])/2, 0)

    x_min, y_min = w_spitzer.all_world2pix(row["ra_max"], row["dec_min"], 0)
    x_max, y_max = w_spitzer.all_world2pix(row["ra_min"], row["dec_max"], 0)
    r = (x_max - x_min)*1

    x_pix_min = x_min - r/2
    y_pix_min = y_min - r/2
    x_pix_max = x_max + r/2
    y_pix_max = y_max + r/2

    if x_pix_min<=0 or y_pix_min<=0:
        return 0, 0
    else:
        c_data = data[int(y_pix_min):int(y_pix_max), int(x_pix_min):int(x_pix_max)].view()
        cut_data = copy.deepcopy(c_data)
        r_shape_y = cut_data.shape[0]
        r_shape_x = cut_data.shape[1]
        cut_data = normalize_rp(cut_data.copy(), 
                                spitzer_rfits.header["PIXSCAL1"], spitzer_gfits.header["PIXSCAL1"])
        
        header['CRVAL1'] = (row["ra_max"] + row["ra_min"])/2
        header['CRVAL2'] = (row["dec_max"] + row["dec_min"])/2
        header['RA'] = (row["ra_max"] + row["ra_min"])/2
        header['DEC'] = (row["dec_max"] + row["dec_min"])/2
        header['CRPIX1'] = cut_data.shape[0]/2
        header['CRPIX2'] = cut_data.shape[1]/2
        header['NAXIS1'] = cut_data.shape[0]
        header['NAXIS2'] = cut_data.shape[1]
        header['CDELT2'] = header['CD2_2']

        new_hdu = astropy.io.fits.PrimaryHDU(cut_data[:,:,0], header)
        new_hdu_list_r = astropy.io.fits.HDUList([new_hdu])
        new_hdu = astropy.io.fits.PrimaryHDU(cut_data[:,:,1], header)
        new_hdu_list_g = astropy.io.fits.HDUList([new_hdu])


        return new_hdu_list_r, new_hdu_list_g


def load_spitzer_fits(spitzer_path):
    spitzer_rfits = astropy.io.fits.open(spitzer_path + 'r.fits')[0]
    spitzer_gfits = astropy.io.fits.open(spitzer_path + 'g.fits')[0]
    spitzer_data = np.concatenate([
        remove_nan(spitzer_rfits.data[:,:,None]),
        remove_nan(spitzer_gfits.data[:,:,None]),
        np.zeros(spitzer_rfits.data.shape)[:,:,None]], axis=2)
    w_spitzer = astropy.wcs.WCS(spitzer_rfits.header)

    return spitzer_rfits, spitzer_gfits, spitzer_data, w_spitzer


def make_momont012_map(fits, w_co, cut_data, vi, v_center=None):
    """
    Moment 0, 1, 2 マップを作成
    
    Args:
        w_co: WCS情報
        cut_data: 3次元データキューブ (velocity, y, x)
        vi: 速度軸の配列
        v_center: 中心速度（この速度を0とする）。Noneの場合は従来通り
    
    Returns:
        dict: moment0, moment1, moment2を含む辞書
    """
    moment1, moment2 = [], []

    dv = abs(fits.header['CDELT3']) / 1000.0
    # Making Moment0 map
    moment0 = np.nansum(cut_data, axis=0) * dv

    # 中心速度が指定されている場合、速度軸をシフト
    if v_center is not None:
        vi_shifted = vi - v_center  # 中心速度を0にシフト
    else:
        vi_shifted = vi

    # Making Moment1+2 map
    for i in range(cut_data.shape[1]):
        for j in range(cut_data.shape[2]):
            I = cut_data[:, i, j]
            
            # Moment 1の計算（シフトした速度を使用）
            aa = I * vi_shifted * dv
            sgm = np.nansum(aa)
            MM = np.nansum(cut_data[:, i, j], axis=0) * dv
            
            if MM > 0:  # ゼロ除算を避ける
                intg = sgm / MM
            else:
                intg = np.nan
                
            moment1.append(intg)

            # Moment 2の計算
            if not np.isnan(intg) and MM > 0:
                bb = I * ((vi_shifted - intg) ** 2) * dv
                cc = np.nansum(bb, axis=0)
                dd = cc / MM
                MM2 = np.sqrt(dd)  # (dd) ** (1/2) をより明確に
            else:
                MM2 = np.nan
                
            moment2.append(MM2)
        
    moment1 = np.array(moment1).reshape(cut_data.shape[1], cut_data.shape[2])
    moment2 = np.array(moment2).reshape(cut_data.shape[1], cut_data.shape[2])

    return {'cut_data': cut_data, 'moment0': moment0, 'moment1': moment1, 'moment2': moment2}