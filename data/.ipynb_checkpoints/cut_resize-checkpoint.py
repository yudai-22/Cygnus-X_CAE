from cut_resize_tools import *
import numpy as np
from astropy.io import fits

vsmooth = 5 #input smoothing kernel size in pix (v direction)~3-5
thresh = 1 #threthould voxel number 200-700
sigma = 1 #threthold value (after smoothing)2-4

sch_rms = 10 #select start ch to calculate the rms (after smoothing)
ech_rms = 90 #select end ch to calculate the rms (after smoothing)

sch_ii=121
ech_ii=241

fits_path = "/home/filament/fujimoto/fits/Cygnus_sp16_vs-40_ve040_dv0.25_12CO_Tmb.fits"

hdu = fits.open(fits_path)[0]
raw_d = hdu.data
head = hdu.header

for pix in [260, 120, 60, 28]:
    cut_data = slide(raw_d[sch_ii:ech_ii], pix)
    print(f"Number of data clipped to {pix} pixels: ", len(cut_data))
    remove_nan_cut_data = remove_nan(cut_data)
    print(f"Number of data after deletion: ", len(remove_nan_cut_data))

    processed_list = parallel_processing(process_data_segment, remove_nan_cut_data, sigma=sigma, 
                                        vsmooth=vsmooth, thresh=thresh, sch_rms=sch_rms, ech_rms=ech_rms)
    processed_list = normalization(processed_list)
    
    print(processed_list[-1].shape, "\n")
    np.save(f"/home/filament/fujimoto/Cygnus-X_CAE/data/zroing_resize_data/original_data/CygnusX_cut_{pix}x{pix}.npy", np.array(processed_list))
    