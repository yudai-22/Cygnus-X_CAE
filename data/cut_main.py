import numpy as np
from astropy.io import fits
from cut_resize_tools import slide, remove_nan, parallel_processing, process_data_segment, normalization, select_conv
from tqdm import tqdm

# Constants
VSmooth = 5  # Input smoothing kernel size in pixels (v direction)
Thresh = 1   # Threshold voxel number
Sigma = 1    # Threshold value (after smoothing)
Sch_RMS = 10  # Start channel to calculate RMS (after smoothing)
Ech_RMS = 90  # End channel to calculate RMS (after smoothing)
Sch_II = 121  # Start index for slicing data
Ech_II = 241  # End index for slicing data

Cut_size_list = [356, 156, 84, 36]
Integrate_layer_num = 30
Obj_size = 100
Obj_sig = 7.5
FITS_PATH = "/home/filament/fujimoto/fits/Cygnus_sp16_vs-40_ve040_dv0.25_12CO_Tmb.fits"
OUTPUT_DIR = "/home/filament/fujimoto/Cygnus-X_CAE/data/zroing_resize_data/original_data/"

def process_fits_data(fits_path, cut_size_list, sch_ii, ech_ii, vsmooth, thresh, sigma, sch_rms, ech_rms, integrate_layer_num, output_dir, obj_size, obj_sig):
    # Load FITS file
    hdu = fits.open(fits_path)[0]
    raw_data = hdu.data
    header = hdu.header  # Header is loaded but unused. Keep if needed for metadata.

    # Iterate over each pixel size
    for pix in cut_size_list:
        print(f"Processing data clipped to {pix} pixels...")
        
        cut_data = slide(raw_data[sch_ii:ech_ii], pix+4)
        print(f"Number of data clipped to {pix} pixels: {len(cut_data)}")
        
        remove_nan_cut_data = remove_nan(cut_data)
        print(f"Number of data after deletion: {len(remove_nan_cut_data)}")
        
        processed_list = parallel_processing(
            process_data_segment, remove_nan_cut_data,
            sigma=sigma, vsmooth=vsmooth, thresh=thresh, sch_rms=sch_rms, ech_rms=ech_rms, 
            integrate_layer_num=integrate_layer_num
        )


        print("Starting convolution")
        conv_list = []
        for _data in tqdm(processed_list):
            _conv_data = select_conv(_data, obj_size, obj_sig)
            conv_list.append(_conv_data)
        
        processed_list = normalization(conv_list)
        
        print(f"Processed data shape for {pix} pixels: {processed_list[-1].shape}")
        
        output_file = f"{output_dir}CygnusX_cut_{pix}x{pix}.npy"
        np.save(output_file, np.array(processed_list))
        print(f"Data saved to {output_file}\n")

def main():
    # Call the processing function with constants
    process_fits_data(
        fits_path=FITS_PATH,
        cut_size_list = Cut_size_list,
        sch_ii=Sch_II,
        ech_ii=Ech_II,
        vsmooth=VSmooth,
        thresh=Thresh,
        sigma=Sigma,
        sch_rms=Sch_RMS,
        ech_rms=Ech_RMS,
        obj_size=Obj_size,
        obj_sig=Obj_sig,
        output_dir=OUTPUT_DIR,
        integrate_layer_num=Integrate_layer_num
    )

# Entry point
if __name__ == "__main__":
    main()