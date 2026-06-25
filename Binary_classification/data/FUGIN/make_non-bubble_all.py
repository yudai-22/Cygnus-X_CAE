import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from astropy.io import fits
from astropy.wcs import WCS
from pathlib import Path
from tqdm import tqdm
from npy_append_array import NpyAppendArray
import gc

import sys
sys.path.append("process_tools/")
from cut_resize_tools import *
from Astronomy import *

# import os
# os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'

# === パラメータ設定 ===
vsmooth = 5
thresh = 1
sigma = 1
sch_rms = 10
ech_rms = 50
sch_ii = 158
ech_ii = 222
percentile = 99.997
width_half = 32
cut_size_list = [100, 50, 25]
integrate_layer_num = 32
obj_size = 128
maximum_mode = "percentile"
cutting_ratio = 2
lower_threshold = 1.66 

# ★メモリ節約のためのバッチサイズ（マシンのスペックに合わせて調整してください）
CHUNK_SIZE = 500 

catalogue_path = '/home/cygnus/fujimoto/Cygnus-X_Molecular_Cloud_Analysis/Bubble_Catalogue/infer_catalogue_all.csv'
catalogue_data = pd.read_csv(catalogue_path)

fits_path_all = glob.glob("/home/cygnus/fujimoto/Cygnus-X_Molecular_Cloud_Analysis/fits/Galaxy_Plane/FUGIN/12CO/FGN_0*00+0000_2x2_12CO_v1.00_cube.fits")
fits_path_all.sort()
integ_path_all = glob.glob("/home/cygnus/fujimoto/Cygnus-X_Molecular_Cloud_Analysis/fits/processed_fits/FUGIN/mom0/FGN_0*00+0000_2x2_12CO_v1.00_cube.pk_vsmooth_2.0_xysmooth_2.0_thresh_700.0_sigma_3.0_mom0.fits")
integ_path_all.sort()

for i in range(len(fits_path_all)):
    fits_path = fits_path_all[i]
    fits_name = fits_path.split("/")[-1]
    region_name = f"{fits_name.split('_')[0]}_{fits_name.split('_')[1]}"
    region_dir = f"./processed_data/{region_name}/thresh_up_low/"
    Path(region_dir).mkdir(parents=True, exist_ok=True)
    
    print("="*25, f"START Region: {region_name}", "="*25)
    
    with fits.open(integ_path_all[i]) as integ_hdu, fits.open(fits_path) as hdu:
        wcs = WCS(integ_hdu[0].header)
        raw_d = hdu[0].data
        
    # --- バブル領域のWCS変換とマスク処理 ---
    # 辞書を使用せず、all_map に直接 NaN を書き込んで簡略化します
    all_map = raw_d.copy()
    
    for index, row in catalogue_data.iterrows():
        # 銀河座標を取得して変換
        world_coords = [
            [row['ra_min'], row['dec_min']], [row['ra_max'], row['dec_min']],
            [row['ra_min'], row['dec_max']], [row['ra_max'], row['dec_max']]
        ]
        region_pix = wcs.wcs_world2pix(world_coords, 0)
        
        # 不正な座標を除外
        valid_pix = [p for p in region_pix if len(p) == 2 and not np.isnan(p).any()]
        if not valid_pix:
            continue
        
        x_coords = [p[0] for p in valid_pix]
        y_coords = [p[1] for p in valid_pix]
        
        x_min, x_max = int(np.floor(np.min(x_coords))), int(np.ceil(np.max(x_coords)))
        y_min, y_max = int(np.floor(np.min(y_coords))), int(np.ceil(np.max(y_coords)))
        
        x_center, y_center = (x_min + x_max) // 2, (y_min + y_max) // 2
        x_range, y_range = (x_max - x_min) // 2 * cutting_ratio, (y_max - y_min) // 2 * cutting_ratio
        
        # 画像範囲を超えないようにクリップ
        x_min, x_max = max(0, int(x_center - x_range)), min(all_map.shape[2], int(x_center + x_range))
        y_min, y_max = max(0, int(y_center - y_range)), min(all_map.shape[1], int(y_center + y_range))
        
        # 領域内でNaNではない要素をNaNにする（元のロジックの踏襲）
        region = all_map[:, y_min:y_max, x_min:x_max]
        region[~np.isnan(region)] = np.nan

    # --- フェーズ1: 座標とメタデータのみを収集 (実データは保持しない) ---
    metadata_list = []
    density_list = []
    velo_width, height, width = all_map.shape
    
    print("Collecting metadata...")
    for pix in cut_size_list:
        step = (pix + 4) // 4
        square = pix + 4
        
        for y in range(0, height - square + 1, step):
            for x in range(0, width - square + 1, step):
                crop = all_map[:, y:y+square, x:x+square]
                
                if np.isnan(crop).any():
                    continue
                
                mean_v = np.nanmean(crop, axis=(1, 2))
                max_v = np.argmax(mean_v)
                start_v = max_v - width_half
                end_v = max_v + width_half
                
                # Z軸の範囲外アクセス防止
                if start_v < 0 or end_v > velo_width:
                    continue
                
                density = np.nanmean(crop[start_v:end_v, :, :])
                density_list.append(density)
                
                metadata_list.append({
                    'x': x, 'y': y, 'square': square,
                    'start_v': start_v, 'end_v': end_v,
                    'density': density
                })

    # --- フェーズ2: プロットの作成 ---
    # ※既存の描画コードをここに配置してください。
    # ★重要: プロット保存後は必ず plt.close() を呼び出してメモリを解放してください。

    # 強度密度の箱ひげ図を作成
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    
    # 1. データの準備
    # single_list が対象の強度データ（1次元リストまたは配列）であると想定します
    data_to_plot = density_list 
    labels = ["All Data"] # グループ名を定義
    
    # --- 統計量の計算 ---
    # NaNを除外して各値を算出します
    q1 = np.nanpercentile(data_to_plot, 25)
    q3 = np.nanpercentile(data_to_plot, 75)
    median = np.nanmedian(data_to_plot)
    mean = np.nanmean(data_to_plot)
    
    # --- 箱ひげ図の描画 ---
    # (plt.figure() は環境に合わせて適宜追加してください)
    
    # tick_labels を使用して Matplotlib 3.9 以降の警告を回避
    box = plt.boxplot(data_to_plot, tick_labels=labels, patch_artist=True, showmeans=True,
                      meanprops={"marker":"s", "markerfacecolor":"magenta", "markeredgecolor":"k"},
                      medianprops={"color": "black", "linewidth": 1}
                      )
    
    # 箱の色を設定（要素が1つなので index 0 を指定）
    box['boxes'][0].set_facecolor("gray")
    box['boxes'][0].set_alpha(0.5)
    
    # 軸とラベルの設定
    plt.yscale('linear') 
    plt.ylabel(r'$\text{Total Intensity (Linear scale)}$')
    plt.title(f'{region_name} Non-bubble Normalized Total Intensity per Bubble of All') 
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    
    # --- 凡例の作成（Q1, Q3 の値を追加） ---
    # 数値は指数表記（.2e）にしています。必要に応じて .2f（小数表記）に変更してください。
    legend_elements = [
        Line2D([0], [0], color='black', lw=1.5, label=f'Median: {median:.3}'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='magenta', markersize=8, label=f'Mean: {mean:.3}'),
        Line2D([0], [0], color='gray', lw=1.5, label=f'Q1 (25%): {q1:.3}'),
        Line2D([0], [0], color='gray', lw=1.5, label=f'Q3 (75%): {q3:.3}'),
    ]
    
    # 凡例を配置（fontsize を調整して重なりを防いでいます）
    plt.legend(handles=legend_elements, loc='upper right', fontsize='small')
    
    # 保存
    plt.savefig(f"Total_Intensity_{region_name}_Boxplot_Stats.png")
    plt.close()

    # 正規化後のプロット
    plt.figure(figsize=(10, 6))
    positive_intensities = np.array(density_list)
    
    # データ数(N)を取得
    N = len(positive_intensities)
    
    # 中央値と平均値の算出
    median_val = np.nanmedian(positive_intensities)
    mean_val = np.nanmean(positive_intensities)
    
    min_val = positive_intensities.min()
    max_val = positive_intensities.max()
    # log_bins = np.logspace(np.log10(min_val), np.log10(max_val), 50)
    line_bins = np.linspace(min_val, max_val, 50)
    
    
    # ヒストグラムの描画
    plt.hist(positive_intensities, bins=line_bins, color="gray", alpha=0.7, edgecolor='w', 
             label=f'Number of Bubbles = ${N}$')
    
    # 中央値の縦線を追加（赤色の破線）
    plt.axvline(median_val, color='black', linestyle='--', linewidth=2, 
                label=f'Median = ${median_val:.2e}$')
    
    # 平均値の縦線を追加（緑色の実線、または点線）
    plt.axvline(mean_val, color='magenta', linestyle='-', linewidth=2, 
                label=f'Mean = ${mean_val:.2e}$')
    
    # 第1四分位数の表示
    plt.axvline(lower_threshold, color='blue', linestyle='-', linewidth=2, 
                label=f'lower threshold = ${lower_threshold:.2e}$')
    
    # # 第3四分位数の表示
    # plt.axvline(upper_threshold, color='red', linestyle='-', linewidth=2, 
    #             label=f'upper threshold = ${upper_threshold:.2e}$')
    
    plt.xlim(0, 10)
    # plt.ylim(0, 60)
    # plt.xscale('log')
    plt.xlabel(r'$\text{Total Intensity (Linear Scale)}$')
    plt.ylabel(r'$\text{Frequency}$')
    plt.title(f'{region_name} Distribution of Normalized Total Intensities for Bubbles')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # 凡例を表示（追加した縦線のラベルも表示されます）
    plt.legend()
    
    # 保存または表示
    plt.savefig(f"{region_name}_Distribution_of_Normalized_Total_Intensities_with_Stats.png")
    plt.close()

    # --- フェーズ3: チャンク単位での本処理と逐次保存 ---
    valid_metadata = [m for m in metadata_list if m['density'] >= lower_threshold]
    print(f"フィルタリング完了: {len(valid_metadata)} 個の配列が残りました。")
    
    # メモリ解放
    del metadata_list, density_list
    gc.collect()

    _, global_max = maximum_value_determination(
        mode=maximum_mode, data=raw_d, vsmooth=vsmooth, sch_rms=sch_rms, ech_rms=ech_rms,
        sch_ii=sch_ii, ech_ii=ech_ii, sigma=sigma, thresh=thresh,
        integrate_layer_num=integrate_layer_num, percentile=percentile
    )

    save_file_path = f"{region_dir}/lower-{lower_threshold}_data_size{obj_size}.npy"
    if os.path.exists(save_file_path):
        os.remove(save_file_path) # 追記時の肥大化を防ぐため初期化
        
    saved_file = NpyAppendArray(save_file_path)

    # チャンクごとに処理を回す
    for idx in tqdm(range(0, len(valid_metadata), CHUNK_SIZE), desc="Processing Chunks"):
        chunk_meta = valid_metadata[idx:idx + CHUNK_SIZE]
        target_arrays = []
        
        # 必要な実データだけを all_map から切り出してリスト化
        for m in chunk_meta:
            crop = all_map[m['start_v']:m['end_v'], m['y']:m['y']+m['square'], m['x']:m['x']+m['square']]
            target_arrays.append(crop)
            
        # 並列処理の実行
        processed_chunk = parallel_processing(
            process_data_segment_necessary, 
            target_arrays,
            vsmooth=vsmooth, sch_rms=sch_rms, ech_rms=ech_rms,
            sigma=sigma, thresh=thresh, integrate_layer_num=integrate_layer_num
        )
        
        # リサイズと正規化
        final_arrays = []
        for _data in processed_chunk:
            _data = gaussian_filter(_data)
            _data = resize(_data, (obj_size, obj_size))
            _data = np.clip(_data, a_min=None, a_max=global_max)
            final_arrays.append(_data)
            
        final_arrays = normalization_thresh(final_arrays, global_max)
        
        # ディスクへ追記保存
        saved_file.append(np.array(final_arrays))
        
        # チャンクごとのメモリ解放
        del target_arrays, processed_chunk, final_arrays
        gc.collect()

    saved_file.close()
    del all_map, raw_d
    gc.collect()
    print("="*25, f"END Region: {region_name}", "="*25)