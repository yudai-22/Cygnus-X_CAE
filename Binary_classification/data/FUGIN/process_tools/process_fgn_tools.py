import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

def process_fugin_fits(fits_path, output_dir, patch_size, stride_ratio, l_range=(10.0, 11.5)):
    """
    FITSファイルから担当範囲内のデータのみを切り出す
    l_range: (min_l, max_l) この範囲に中心があるデータのみ採用する
    """
    with fits.open(fits_path) as hdul:
        data = hdul[0].data
        header = hdul[0].header
        wcs = WCS(header)

        # 例: patch_size x patch_sizeでスライド窓切り出しを行う
        patch_size = patch_size
        stride = patch_size // stride_ratio # ストライド
        
        ny, nx = data.shape[-2:] # 速度軸がある場合は後ろ2つを指定
        
        count = 0
        for y in range(0, ny - patch_size, stride):
            for x in range(0, nx - patch_size, stride):
                # 1. 切り出し領域の中心ピクセル座標
                center_x = x + patch_size / 2
                center_y = y + patch_size / 2
                
                # 2. ピクセル座標を天球座標(l, b)に変換
                # wcs.pixel_to_world もしくは wcs.all_pix2world を使用
                # FUGINは通常 (l, b, v) なので、適切な軸を指定
                lon, lat, _ = wcs.all_pix2world(center_x, center_y, 0, 0)
                
                # 3. 担当範囲内かチェック (l_range)
                if l_range[0] <= lon < l_range[1]:
                    # 範囲内なら切り出し処理を実行
                    patch = data[:, y:y+patch_size, x:x+patch_size]
                    
                    count += 1
        
        print(f"{fits_path}: {count} patches created in range {l_range}")