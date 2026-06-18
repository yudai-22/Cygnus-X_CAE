from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
import aplpy

import pandas as pd
import sys
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import glob
from tqdm.notebook import tqdm

sys.path.append("process_tools/")
from cut_resize_tools import *
from Astronomy import *


def process_data_segment_necessary(data, vsmooth, sch_rms, ech_rms, sigma, thresh, integrate_layer_num):
    _data = data.copy()
    vconv = convolve_vaxis(_data, vsmooth)
    _mask = vconv.copy()
    rms = np.nanstd(vconv[sch_rms:ech_rms], axis=0)
    _mask[np.where(_mask < rms*sigma)] = 0
    ndata, _ = picking(_mask, data, thresh)      
    ndata = integrate_to_x_layers(ndata, integrate_layer_num)
    return ndata


vsmooth = 5
thresh = 1
sigma = 1
sch_rms = 10
ech_rms = 50
sch_ii = 158
ech_ii = 222
percentile = 99.997
width_half = 32
sigma_multiply = None
cut_size_list = np.array([100, 50, 25])
integrate_layer_num = 32
obj_size = 128
maximum_mode = "percentile"

cutting_ratio = 2
lower_threshold = 1.66 


catalogue_path = '/home/cygnus/fujimoto/Cygnus-X_Molecular_Cloud_Analysis/Bubble_Catalogue/infer_catalogue_all.csv' # 提供されたCSVファイル
catalogue_data = pd.read_csv(catalogue_path)


fits_path_all = glob.glob("/home/cygnus/fujimoto/Cygnus-X_Molecular_Cloud_Analysis/fits/Galaxy_Plane/FUGIN/12CO/FGN_0[8-9]00+0000_2x2_12CO_v1.00_cube.fits")
fits_path_all.sort()
integ_path_all = glob.glob("/home/cygnus/fujimoto/Cygnus-X_Molecular_Cloud_Analysis/fits/processed_fits/FUGIN/mom0/FGN_01[8-9]00+0000_2x2_12CO_v1.00_cube.pk_vsmooth_2.0_xysmooth_2.0_thresh_700.0_sigma_3.0_mom0.fits")
integ_path_all.sort()

for i in range(len(fits_path_all)):
    fits_path = fits_path_all[i]
    integ_hdu = fits.open(integ_path_all[i])[0]
    
    w_high = WCS(fits_path)
    hdu = fits.open(fits_path)[0]
    wcs = WCS(integ_hdu.header)
    
    raw_d = hdu.data
    header = hdu.header
    
    
    # fitsのPathからfitsの名前のみを取り出す
    fits_name = fits_path.split("/")[-1]
    region_name = fits_name.split("_")
    region_name = region_name[0] + "_" + region_name[1]
    region_dir = "./processed_data/" + region_name + "/thresh_up_low/"
    print("="*25, f"START Region: {region_name}", "="*25)
    
    
    # バブル座標をすべてピクセル情報に変換するセル
    
    babble_region_galactic = [] # 銀河座標を格納するリスト
    for index, row in catalogue_data.iterrows():
        glon_min = row['ra_min'] # カタログにはなぜかra, decの場所にglon, glatの座標が入っている
        glon_max = row['ra_max']
        glat_min = row['dec_min']
        glat_max = row['dec_max']
    
        # (l, b) のリストを作成
        babble_region_galactic.append([
            [glon_min, glat_min], # 0: L_min, B_min　右下
            [glon_max, glat_min], # 1: L_max, B_min　左下
            [glon_min, glat_max], # 2: L_min, B_max　右上
            [glon_max, glat_max]  # 3: L_max, B_max　左上
        ])
    print(f"babble_region_galactic len: {len(babble_region_galactic)}")
    
    # --- babble_region_pix への変換 (修正) ---
    babble_region_pix = []
    for i in range(len(babble_region_galactic)):
        babble_region = babble_region_galactic[i] # 銀河座標のバブル領域
        region_list = []
        for j, world_coords_list in enumerate(babble_region):
            world_coords_list = [babble_region[j]] # [L, B] のリスト
    
            # WCS変換。世界座標 (L, B) をピクセル座標に変換
            # origin=0 でPythonの0-based indexに対応
            region_pix_result = wcs.wcs_world2pix(world_coords_list, 0)
    
            # 変換結果の形状と要素数をチェックする
            if len(region_pix_result) > 0 and len(region_pix_result[0]) == 2:
                region_list.append(region_pix_result[0]) # 期待通り [x, y] を追加
            else:
                print(f"警告: bubble_num={i}, point_idx={j} のWCS変換結果が不正です: {region_pix_result}. NaNで埋めます。")
                region_list.append([np.nan, np.nan]) # NaNで埋めて、後で除外されるようにする
    
        babble_region_pix.append(region_list)
    print(f"babble_region_pix len: {len(babble_region_pix)}")
    
    
    # raw_dからバブルを切り取る
    cutting_data_dic = {}
    data_to_cut = raw_d.copy() # raw_d からのコピー
    
    for bubble_num in range(len(babble_region_pix)):
        all_x_coords = [p[0] for p in babble_region_pix[bubble_num]]
        all_y_coords = [p[1] for p in babble_region_pix[bubble_num]]
    
        # ピクセル座標の最小値と最大値を取得
        # np.min と np.max を使うことで、座標の順序に関わらず正しい矩形領域の境界を取得
        # これにより、x_max_pix < x_min_pix のような逆転を防ぎ、スライス範囲が巨大になるのを防ぐ
        x_min_pix = int(np.floor(np.min(all_x_coords)))
        x_max_pix = int(np.ceil(np.max(all_x_coords)))
        y_min_pix = int(np.floor(np.min(all_y_coords)))
        y_max_pix = int(np.ceil(np.max(all_y_coords)))
    
        x_center = (x_min_pix + x_max_pix) // 2 
        y_center = (y_min_pix + y_max_pix) // 2
        
        x_range = (x_max_pix - x_min_pix) // 2 * cutting_ratio
        y_range = (y_max_pix - y_min_pix) // 2 * cutting_ratio
    
        x_min_pix = int(x_center - x_range)
        x_max_pix = int(x_center + x_range)
        y_min_pix = int(y_center - y_range)
        y_max_pix = int(y_center + y_range)
    
        # 切り出し実行
        # ここでは、np.clip を使用せず、NumPyのスライスが範囲外を0で自動パディングする動作に任せる
        # ただし、x_min_pix/y_min_pix が負になったり、x_max_pix/y_max_pix が大きくなりすぎたりすると、
        # 巨大な配列が生成される可能性がある点に注意（メモリ消費、処理時間）
        cutting_data = data_to_cut[:, y_min_pix:y_max_pix, x_min_pix:x_max_pix]
        pix_cood = (x_min_pix, x_max_pix, y_min_pix, y_max_pix)
        cutting_data_dic[pix_cood] = cutting_data
    
        # デバッグ用に切り出したデータの形状と範囲を表示してみる
        # print(f"Bubble {bubble_num}: X range [{x_min_pix}, {x_max_pix}], Y range [{y_min_pix}, {y_max_pix}], Shape: {cutting_data.shape}")
    
    
    deleted_indices = []
    filtered_list = [
        data for idx, data in enumerate(cutting_data_dic.items()) 
        if not np.all((data[1] == 0) | np.isnan(data[1]))
        or deleted_indices.append(idx)  # 削除されたインデックスを記録
    ]
    
    all_map = raw_d.copy()
    
    for pdu in filtered_list:
        pdu = pdu[0]
        x_min, x_max, y_min, y_max = pdu[0], pdu[1], pdu[2], pdu[3]
    
        # 1. 対象領域をスライスで取得
        region = all_map[:, y_min:y_max, x_min:x_max]
        
        # 2. 領域内でNaNではない要素のマスクを作成
        # np.isnan(region) は NaN の位置が True になる
        # ~ で論理反転させ、NaNで「ない」位置を True にする
        not_nan_mask = ~np.isnan(region)
        
        # 3. マスクがTrueの位置（NaNでない要素）だけを0にする
        region[not_nan_mask] = np.nan
    
    # データの切り取り
    cut_all_list = []
    for pix in cut_size_list:
        print(f"Processing data clipped to {pix} pixels...")
    
        # Step1: スライス
        cut_data = slide(all_map, pix+4, 4)
        print(f"Number of data clipped to {pix} pixels: {len(cut_data)}")
    
        cut_data = remove_nan(cut_data)
        print(f"Number of data after deletion: {len(cut_data)}")
    
        cut_all_list += cut_data
    
    # 使用チャンネルの決定
    proccessed_v_list = []
    for data in cut_all_list:
        mean = np.nanmean(data.copy(), axis=(1, 2))
        max_intens_v = np.argmax(mean)
        start_v = max_intens_v - width_half
        end_v = max_intens_v + width_half
        proccessed_v_list.append(data[start_v:end_v, :, :])
    
    del cut_all_list
    
    # データの強度密度計算
    density_list = []
    
    for arr in proccessed_v_list:
        # 3次元配列内の NaN を無視して平均値（1pixあたりの強度）を計算
        # これにより (合計強度) / (有効なピクセル数) が算出されます
        density = np.nanmean(arr)
        density_list.append(density)
    
    
    print(f"計算完了: {len(density_list)} 個の密度を算出しました。")
    
    
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
    plt.show()
    
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
    plt.show()
    
    
    # 閾値の条件を満たす配列だけを抽出する
    # array_list: 元の3次元配列のリスト
    # density_list: 前のステップで計算した1pixあたりの平均値のリスト
    
    filtered_array_list = []
    filtered_density_list = []
    
    for arr, dens in zip(proccessed_v_list, density_list):
        # 平均値（密度）がしきい値の範囲内にあるか判定
        if lower_threshold <= dens:
        # <= upper_threshold:
            filtered_array_list.append(arr)
            filtered_density_list.append(dens)
        else:
            # 範囲外の場合は新リストに追加しない（＝削除と同じ結果）
            # メモリを即座に解放したい場合は、ここで個別の配列を消去する
            del arr
    
    # 3. 不要になった元のリストを削除してメモリを解放
    del proccessed_v_list
    del density_list
    
    import gc
    gc.collect()
    
    print(f"フィルタリング完了: {len(filtered_array_list)} 個の配列が残りました。")
    
    # Step2: 並列前処理
    processed_list = parallel_processing(
                                        function=process_data_segment_necessary, 
                                        target=filtered_array_list,
                                        vsmooth=vsmooth,
                                        sch_rms=sch_rms, 
                                        ech_rms=ech_rms,
                                        sigma=sigma, 
                                        thresh=thresh, 
                                        integrate_layer_num=integrate_layer_num
                                        )
    
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
    
    conv_resize_list = []
    for _data in tqdm(processed_list):
        _data = gaussian_filter(_data)
        _data = resize(_data, (obj_size, obj_size))
        if maximum_mode in ["percentile", "sigma"]:
            _data = np.clip(_data, a_min=None, a_max=max_thresh)
        conv_resize_list.append(_data)
    del processed_list
    
    conv_resize_list = normalization_thresh(conv_resize_list, max_thresh)
    
    Path(region_dir).mkdir(parents=True, exist_ok=True)
    
    np.save(f"./{region_dir}/lower-1.66_data_size128", conv_resize_list)