from astropy.io import fits
from astropy.wcs import WCS
import astropy.units as u
import aplpy

import pandas as pd
import sys
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import glob
from tqdm import tqdm

sys.path.append("process_tools/")
from cut_resize_tools import *
from Astronomy import *

# ==========================================
# パラメータ設定
# ==========================================
vsmooth = 5
thresh = 10
sigma = 1
sch_rms = 10
ech_rms = 50
sch_ii = 160
ech_ii = 220
percentile = 99.997
width_half = 30
sigma_multiply = None
cut_size_list = np.array([100, 50, 25])
integrate_layer_num = 30
obj_size = 100
maximum_mode = "percentile"
cutting_ratio = 2

# ==========================================
# パス設定とデータの読み込み
# ==========================================
catalogue_path = '/home/cygnus/fujimoto/Cygnus-X_Molecular_Cloud_Analysis/Bubble_Catalogue/infer_catalogue_all.csv' 
catalogue_data = pd.read_csv(catalogue_path)

fits_path_all = glob.glob("/home/cygnus/fujimoto/Cygnus-X_Molecular_Cloud_Analysis/fits/Galaxy_Plane/FUGIN/12CO/*.fits")
integ_fits_path_all = glob.glob("/home/cygnus/fujimoto/Cygnus-X_Molecular_Cloud_Analysis/fits/processed_fits/FUGIN/mom0/*")

# リストのソート（タイポ .dort() を修正）
fits_path_all.sort()
integ_fits_path_all.sort()

# テスト用に最初の8つのみ取得
fits_path_all = fits_path_all[:8]
integ_fits_path_all = integ_fits_path_all[:8]
total_fits = len(fits_path_all)

# ==========================================
# メイン処理ループ
# ==========================================
for fits_num, (fits_path, integ_fits_path) in enumerate(zip(fits_path_all, integ_fits_path_all)):
    
    # --- [追加機能] 現在処理中のfitsファイルを視覚的にわかりやすく表示 ---
    fits_name = fits_path.split("/")[-1]
    print("\n" + "="*60)
    print(f"🚀 Processing FITS [{fits_num + 1}/{total_fits}]: {fits_name}")
    print("="*60 + "\n")
    
    # 保存先ディレクトリの設定
    region_name_parts = fits_name.split("_")
    region_name = region_name_parts[0] + "_" + region_name_parts[1]
    region_dir = f"./processed_data/{region_name}"
    picture_dir = f"./data_pictures/{region_name}"
    print(f"📂 Output Directory: {region_dir}")
    
    # データの読み込み
    hdu = fits.open(fits_path)[0]
    integ_hdu = fits.open(integ_fits_path)[0]
    
    wcs = WCS(fits_path)
    integ_wcs = WCS(integ_fits_path)
    
    raw_d = hdu.data
    header = hdu.header
    
    # ------------------------------------------
    # 1. カタログからバブル領域の銀河座標を取得
    # ------------------------------------------
    bubble_region_galactic = []
    for index, row in catalogue_data.iterrows():
        # カタログのra, dec列にglon, glatが格納されている想定
        glon_min, glon_max = row['ra_min'], row['ra_max']
        glat_min, glat_max = row['dec_min'], row['dec_max']
    
        bubble_region_galactic.append([
            [glon_min, glat_min], # L_min, B_min
            [glon_max, glat_min], # L_max, B_min
            [glon_min, glat_max], # L_min, B_max
            [glon_max, glat_max]  # L_max, B_max
        ])
    
    # ------------------------------------------
    # 2. 銀河座標(L, B)をピクセル座標へ変換
    # ------------------------------------------
    bubble_region_pix = []
    for i, bubble_region in enumerate(bubble_region_galactic):
        region_list = []
        for j, world_coords in enumerate(bubble_region):
            # WCS変換。origin=0でPythonの0-based indexに対応
            region_pix_result = integ_wcs.wcs_world2pix([world_coords], 0)
    
            if len(region_pix_result) > 0 and len(region_pix_result[0]) == 2:
                region_list.append(region_pix_result[0]) # [x, y] を追加
            else:
                print(f"⚠️ 警告: bubble_num={i}, point_idx={j} のWCS変換結果が不正です: {region_pix_result}. NaNで埋めます。")
                region_list.append([np.nan, np.nan])
    
        bubble_region_pix.append(region_list)
    
    # ------------------------------------------
    # 3. ピクセル座標からバブル領域を切り出す
    # ------------------------------------------
    cutting_data_dic = {}
    data_to_cut = raw_d.copy()
    
    for bubble_num, coords in enumerate(bubble_region_pix):
        all_x_coords = [p[0] for p in coords]
        all_y_coords = [p[1] for p in coords]
    
        x_min_pix, x_max_pix = int(np.floor(np.min(all_x_coords))), int(np.ceil(np.max(all_x_coords)))
        y_min_pix, y_max_pix = int(np.floor(np.min(all_y_coords))), int(np.ceil(np.max(all_y_coords)))
    
        # 中心座標と切り出し範囲の計算
        x_center = (x_min_pix + x_max_pix) // 2 
        y_center = (y_min_pix + y_max_pix) // 2
        
        x_range = ((x_max_pix - x_min_pix) // 2) * cutting_ratio
        y_range = ((y_max_pix - y_min_pix) // 2) * cutting_ratio
    
        x_min_pix, x_max_pix = int(x_center - x_range), int(x_center + x_range)
        y_min_pix, y_max_pix = int(y_center - y_range), int(y_center + y_range)
    
        # Numpyのスライス（範囲外は自動で処理される）
        cutting_data = data_to_cut[:, y_min_pix:y_max_pix, x_min_pix:x_max_pix]
        pix_cood = (x_min_pix, x_max_pix, y_min_pix, y_max_pix)
        cutting_data_dic[pix_cood] = cutting_data
    
    # ------------------------------------------
    # 4. 無効なデータのフィルタリングと元マップのマスク
    # ------------------------------------------
    deleted_indices = []
    filtered_list = []
    for idx, (coord, data) in enumerate(cutting_data_dic.items()):
        # すべてが0 または NaN のデータを除外
        if not np.all((data == 0) | np.isnan(data)):
            filtered_list.append((coord, data))
        else:
            deleted_indices.append(idx)
    
    # print(f"🗑️ 削除されたデータのインデックス: {deleted_indices}")
    print(f"📉 削除された数: {len(deleted_indices)} | 残ったデータ数: {len(filtered_list)}/{len(catalogue_data)}")
    
    all_map = raw_d.copy()
    
    # マスク処理（バブル領域をNaNにする）
    for pdu in filtered_list:
        coord = pdu[0]
        x_min, x_max, y_min, y_max = coord[0], coord[1], coord[2], coord[3]
    
        region = all_map[:, y_min:y_max, x_min:x_max]
        not_nan_mask = ~np.isnan(region)
        region[not_nan_mask] = np.nan
    
    # ------------------------------------------
    # 5. バブル除外後のデータから学習用データを切り抜き・前処理
    # ------------------------------------------
    processed_all_list = []
    for pix in cut_size_list:
        print(f"⏳ Processing data clipped to {pix} pixels...")
        
        cut_data = slide(all_map, pix+4, 4)
        print(f"   Number of data clipped to {pix} pixels: {len(cut_data)}")
    
        cut_data = remove_nan(cut_data)
        print(f"   Number of data after NaN deletion: {len(cut_data)}")
    
        # 並列前処理
        processed_list = parallel_processing(
            function=process_data_segment_center_ver, 
            target=cut_data,
            vsmooth=vsmooth,
            sch_rms=sch_rms, 
            ech_rms=ech_rms,
            sigma=sigma, 
            thresh=thresh, 
            width=width_half,
            integrate_layer_num=integrate_layer_num
        )
        processed_all_list += processed_list
    
    # 最大値の決定
    max_thresh = maximum_value_determination(
         mode=maximum_mode, 
         data=raw_d, 
         vsmooth=vsmooth, 
         sch_rms=sch_rms, 
         ech_rms=ech_rms, 
         sch_ii=sch_ii, 
         ech_ii=ech_ii, 
         sigma=sigma, 
         thresh=thresh, 
         integrate_layer_num=integrate_layer_num, 
         percentile=percentile
    )
    
    # ------------------------------------------
    # 6. リサイズ、クリッピング、正規化
    # ------------------------------------------
    conv_resize_list = []
    for _data in tqdm(processed_all_list, desc="Resizing and Filtering"):
        _data = gaussian_filter(_data)
        _data = resize(_data, (obj_size, obj_size))
        
        if maximum_mode in ["percentile", "sigma"]:
            _data = np.clip(_data, a_min=None, a_max=max_thresh)
        conv_resize_list.append(_data)
    del processed_all_list
    
    conv_resize_list = normalization_thresh(conv_resize_list, max_thresh)
    
    # ------------------------------------------
    # 7. 確認用のプロットと保存
    # ------------------------------------------
    # プロット数がデータ数より多い場合にエラーになるのを防ぐため最小値を取る
    plot_num = min(20, len(conv_resize_list))
    if plot_num > 0:
        # 重複なしでランダムにインデックスを取得
        random_num = sorted(np.random.choice(len(conv_resize_list), plot_num, replace=False))
        print(f"📊 Plotting random samples: {random_num}")
        
        axes_num = conv_resize_list[0].shape[0]
        Path(picture_dir).mkdir(parents=True, exist_ok=True) # 保存先ディレクトリの確保
        
        # --- (A) チャンネルごとのマップ保存 ---
        for index in random_num:
            data = conv_resize_list[index]
            
            fig, axes = plt.subplots(1, axes_num, figsize=(axes_num*3, 1*4))
            if axes_num == 1:
                axes = [axes]

            for j, data_segment in enumerate(data):
                axes[j].imshow(data_segment, vmin=0, vmax=1, cmap="jet")
                axes[j].axis("off")
                axes[j].set_title(f"{j+1}ch", fontsize=20, color='red')
        
            fig.suptitle(f"Data No.{index}, min={data.min():.2f}, max={data.max():.2f}", fontsize=30) 
            plt.tight_layout()
            plt.savefig(f"{picture_dir}/{region_name}_random_channelmap_No{index}.png")
            plt.close(fig) # 保存したらすぐに閉じてメモリ解放
            
        # --- (B) 積分強度図の保存（追加機能） ---
        print("🌌 Saving integrated intensity maps...")
        fig_int, axes_int = plt.subplots(2, 10, figsize=(3*10, 6))
        axes_int_flat = axes_int.flatten() # ループの前に1度だけ1次元化する
        
        for i, num in enumerate(random_num):
            data = conv_resize_list[num]
            # 積分処理
            integ_data = np.nansum(data, axis=0)

            axes_int_flat[i].imshow(integ_data, vmin=0, vmax=1, cmap="cividis", ) # お好みでcmapを指定
            axes_int_flat[i].axis("off")
            axes_int_flat[i].set_title(f"cut_number {num}")

        # データが20個未満で余ったサブプロットがあれば非表示にする
        for j in range(plot_num, len(axes_int_flat)):
            axes_int_flat[j].axis("off")

        plt.tight_layout()
        plt.savefig(f"{picture_dir}/{region_name}_integrated_random_data.png")
        plt.close(fig_int) # こちらも保存後にメモリ解放
    
    # 最終データの保存
    np.save(f"{region_dir}/non_bubble_data", conv_resize_list)
    print(f"✅ Saved completely to {region_dir}/non_bubble_data.npy")
    print(f"✅ Saved completely to {region_dir}/non_bubble_data.npy")