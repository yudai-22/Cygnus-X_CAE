from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
import aplpy

import pandas as pd
import sys
from pathlib import Path
import glob
import numpy as np
import matplotlib.pyplot as plt
import itertools

import torch

sys.path.append("/home/cygnus/fujimoto/Cygnus-X_Molecular_Cloud_Analysis/Binary_classification/data/FUGIN/process_tools")
from cut_resize_tools import *
from Astronomy import *


fits_paths = glob.glob("/home/cygnus/fujimoto/Cygnus-X_Molecular_Cloud_Analysis/fits/Galaxy_Plane/FUGIN/12CO/FGN_01600+0000_2x2_12CO_v1.00_cube.fits")
fits_paths.sort()

integ_fits_paths = glob.glob("/home/cygnus/fujimoto/Cygnus-X_Molecular_Cloud_Analysis/fits/processed_fits/FUGIN/mom0/FGN_01600+0000_2x2_12CO_v1.00_cube.pk_vsmooth_2.0_xysmooth_2.0_thresh_700.0_sigma_3.0_mom0.fits")
integ_fits_paths.sort()

rank_csv_paths = glob.glob("/home/cygnus/fujimoto/Cygnus-X_Molecular_Cloud_Analysis/Bubble_Catalogue/rank_catalogue/MWP_Rank*.csv")
rank_csv_paths.sort()


# データ処理に関する定数の設定
cutting_ratio = 2 # バブルの切り取り比率
vsmooth = 5 # 速度方向のスムース倍率
thresh = 1 # pickingでオブジェクトとみなすpix数
sigma = 1 # 
sch_rms = 10
ech_rms = 50
sch_ii = 160
ech_ii = 220
percentile = 99.997
width_half = 30
sigma_multiply = None
integrate_layer_num = 30
obj_size = 100
maximum_mode = "percentile"

# fits毎にRank1~3のバブルデータを作成していく
for fits_num in range(len(fits_paths)):
    fits_path = fits_paths[fits_num]
    integ_fits_path = integ_fits_paths[fits_num]

    hdu = fits.open(fits_path)[0]
    integ_hdu = fits.open(integ_fits_path)[0]
    wcs = WCS(integ_hdu.header)
    raw_d = hdu.data
    header = hdu.header

    for rank in range(len(rank_csv_paths)):
        rank_csv = pd.read_csv(rank_csv_paths[rank])
        rank += 1

        # 画像を保存するディレクトの準備
        fits_name = fits_path.split("/")[-1]
        region_name = fits_name.split("_")
        region_name = region_name[0] + "_" + region_name[1]
        region_dir = "./processed_data/" + region_name + "/Rank_Bubble" + f"/Rank{rank}"
        
        line_decoration = "=" * 20
        print(f"{line_decoration} Start Processing. region:{region_name}, Rank:{rank} {line_decoration}")
        Path(region_dir).mkdir(parents=True, exist_ok=True)


        ny, nx = raw_d.shape[1:3]
        glon_min, glat_min = wcs.all_pix2world(nx, 0, 0)
        glon_max, glat_max = wcs.all_pix2world(0, ny, 0)
        
        # fitsが担当する領域を調整
        #  FGN_01100➡️l=10°~12°→10°~11.5°
        #  FGN_01200➡️l=11°~13°→11.5°~12.5°
        #  FGN_01300➡️l=12°~14°→12.5°~13.5°
        # ……
        
        if glon_min > 10:
            glon_min += 0.5
        glon_max -= 0.5
        
        print("glon_min: ", glon_min,"\n",
              "glon_max: ", glon_max,"\n",
              "glat_min: ", glat_min,"\n",
              "glat_max: ", glat_max)
        
        rank_1_catalog = 0
        rank_2_catalog = 0
        rank_3_catalog = 0
        rank_catalog_list = [rank_1_catalog, rank_2_catalog, rank_3_catalog]
        
        catalogue_data = rank_csv
        catalogue_data_selected = catalogue_data.query(
        f"{glon_min} <= GLON and GLON <= {glon_max} and {glat_min} <= GLAT and GLAT <= {glat_max}"
        ).reset_index()

        # 範囲内のバブルがゼロだった場合ループをスキップ
        if len(catalogue_data_selected) == 0:
            print(f"{region_name} Rank:{rank} Bubble　is None. \n This process is skipped" )
            continue

        GLON_center = catalogue_data_selected["GLON"]
        GLAT_center = catalogue_data_selected["GLAT"]
        Reff = catalogue_data_selected["Reff"] / 60
        
        glon_min = GLON_center - Reff
        glon_max = GLON_center + Reff
        glat_min = GLAT_center - Reff
        glat_max = GLAT_center + Reff
        
        catalogue_data_selected["glon_min"] = glon_min
        catalogue_data_selected["glon_max"] = glon_max
        catalogue_data_selected["glat_min"] = glat_min
        catalogue_data_selected["glat_max"] = glat_max

        babble_region_galactic = [] # 銀河座標を格納するリスト
        for index, row in catalogue_data_selected.iterrows():
            glon_min = row['glon_min'] # カタログにはなぜかra, decの場所にglon, glatの座標が入っている
            glon_max = row['glon_max']
            glat_min = row['glat_min']
            glat_max = row['glat_max']
        
            # (l, b) のリストを作成
            babble_region_galactic.append([
                [glon_min, glat_min], # 0: L_min, B_min
                [glon_max, glat_min], # 1: L_max, B_min
                [glon_min, glat_max], # 2: L_min, B_max
                [glon_max, glat_max]  # 3: L_max, B_max
            ])
        print(f"babble_region_galactic len: {len(babble_region_galactic)}")
        
        # --- babble_region_pix への変換 (修正) ---
        # ループ元を babble_region_galactic に変更
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

        # データにの切り取り
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
            cutting_data_sahape = cutting_data.shape
            
            # 4pix以下の形状を持つデータはガウシアンフィルターでopixとなってしまうため、保存しない。
            if any(s <= 4 for s in cutting_data_sahape) :
               # デバッグ用に切り出したデータの形状と範囲を表示してみる
                print(f"Bubble {bubble_num}: X range [{x_min_pix}, {x_max_pix}], Y range [{y_min_pix}, {y_max_pix}], Shape: {cutting_data.shape}")
                print("deleted !!") 
            else:
                pix_cood = (x_min_pix, x_max_pix, y_min_pix, y_max_pix)
                cutting_data_dic[pix_cood] = cutting_data
                # デバッグ用に切り出したデータの形状と範囲を表示してみる
                print(f"Bubble {bubble_num}: X range [{x_min_pix}, {x_max_pix}], Y range [{y_min_pix}, {y_max_pix}], Shape: {cutting_data.shape}")


        # エラーデータの削除
        filtered_list = []
        deleted_indices = []
        
        for idx, (key, value) in enumerate(cutting_data_dic.items()):
            # 0が含まれる、または、NaNが含まれるかチェック
            if np.any(value == 0) or np.any(np.isnan(value)):
                deleted_indices.append(idx)
            else:
                filtered_list.append((key, value))
        
        # 削除されたインデックスを表示
        print("削除されたデータのインデックス:", deleted_indices)
        print("削除された数: ", len(deleted_indices))
        print("残ったデータ数: ",len(filtered_list))

        all_map = raw_d.copy()

        # オーグメンテション用に移動倍率リストの作成
        slide_val_list = torch.tensor([-0.25, 0, 0.25])
        result = torch.cartesian_prod(slide_val_list, slide_val_list)

        # データと使用速度を結びつける
        start_end_no_aug_list = []
        start_end_list = []
        
        for n in range(len(filtered_list)): 
            data = filtered_list[n][1]
            mean = np.nanmean(data, axis=(1, 2))
            max_intens_v = np.argmax(mean)
            start_v = max_intens_v - width_half
            end_v = max_intens_v + width_half
            start_end_no_aug_list.append((start_v, end_v))
        
            for i in range(len(result)):
                start_end_list.append((start_v, end_v))

        slide_list = []
        for idx, pdu in enumerate(filtered_list):
            idx *= 9
            pdu = pdu[0]
            x_min, x_max, y_min, y_max = pdu[0], pdu[1], pdu[2], pdu[3]
            diff_x = x_max - x_min
            diff_y = y_max - y_min
        
            for slide_val_x, slide_val_y in result:
                x_min_s = int(x_min + diff_x*slide_val_x)
                x_max_s = int(x_max + diff_x*slide_val_x)
                y_min_s = int(y_min + diff_y*slide_val_y)
                y_max_s = int(y_max + diff_y*slide_val_y)
        
                region = all_map[:, y_min_s:y_max_s, x_min_s:x_max_s].copy()
                slide_list.append([region, start_end_list[idx]])
                idx += 1

        deleted_indices = []
        filterd_list = []
        start_end_list = []
        
        for idx, data in enumerate(slide_list):
            if np.all((data[0] == 0) | np.isnan(data[0])):
                deleted_indices.append(idx)
                data[0] = np.zeros((462, 100, 100))
        
            filterd_list.append(data[0])
            start_end_list.append(data[1])
        
        # 削除されたインデックスを表示
        print("ゼロで埋められたデータのインデックス:", deleted_indices)
        print("ゼロで埋られた数: ", len(deleted_indices))
        print("データ数: ",len(filterd_list))


        processed_list = parallel_processing_advanced(
                                        function=process_data_segment_set_StartEnd, 
                                        target_list=filterd_list,
                                        variable_list=start_end_list,
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

        channel_map_list = []
        for coordinate, data in filtered_list:
            channel_map_list.append(data)

        processed_no_aug_list = parallel_processing_advanced(
                                        function=process_data_segment_set_StartEnd, 
                                        target_list=channel_map_list,
                                        variable_list=start_end_no_aug_list,
                                        vsmooth=vsmooth,
                                        sch_rms=sch_rms,
                                        ech_rms=ech_rms,
                                        sigma=sigma,
                                        thresh=thresh,
                                        integrate_layer_num=integrate_layer_num
                                        )

        # smoothingとresizeと正規化
        conv_resize_list = []
        for _data in tqdm(processed_list):
            # _data = select_conv(_data, obj_size, obj_sig)
            _data = gaussian_filter(_data)
            _data = resize(_data, (obj_size, obj_size))
            if maximum_mode in ["percentile", "sigma"]:
                _data = np.clip(_data, a_min=None, a_max=max_thresh)
            conv_resize_list.append(_data)
        
        conv_resize_list = normalization_thresh(conv_resize_list, max_thresh)

        conv_resize_no_aug_list = []
        for _data in tqdm(processed_no_aug_list):
            # _data = select_conv(_data, obj_size, obj_sig)
            _data = gaussian_filter(_data)
            _data = resize(_data, (obj_size, obj_size))
            if maximum_mode in ["percentile", "sigma"]:
                _data = np.clip(_data, a_min=None, a_max=max_thresh)
            conv_resize_no_aug_list.append(_data)
        
        conv_resize_no_aug_list = normalization_thresh(conv_resize_no_aug_list, max_thresh)

        #データの画像保存
        vals = [-0.25, 0, 0.25]
        combinations = [list(x) for x in itertools.product(vals, repeat=2)]
        combinations *= len(conv_resize_no_aug_list)

        # オーグメンテーションされたデータの積分強度図の確認
        ##プロット用のデータを一時的に蓄えるリスト
        current_batch_indices = []
        set_count = 0  # セット番号のカウンター
        
        # 2D画像保存ディレクトリ準備
        rank_dir_name = f"{region_dir}/2D_map"
        Path(rank_dir_name).mkdir(parents=True, exist_ok=True)
        
        for i, comb in enumerate(combinations):
            current_batch_indices.append(i)
        
            # 条件 [0.25, 0.25] に一致するか、または全データの最後かチェック
            if list(comb) == [0.25, 0.25] or i == len(combinations) - 1:
                
                num_plots = len(current_batch_indices)
                fig, axes = plt.subplots(1, num_plots, figsize=(2 * num_plots, 3)) # タイトル用に高さを少し広げました
                
                if num_plots == 1:
                    axes = [axes]
                else:
                    axes = axes.flatten()
        
                for j, idx in enumerate(current_batch_indices):
                    ax = axes[j]
                    data = conv_resize_list[idx]
                    data = np.sum(data, axis=0)
        
                    ax.imshow(data, cmap="viridis") 
                    ax.set_title(f"{combinations[idx]}")
                    ax.axis("off")
        
                # --- Figure全体へのタイトル設定 ---
                fig.suptitle(f"No.{set_count}, Data shape: {processed_list[idx].shape[1:]}", fontsize=16, fontweight='bold')
                # --------------------------------
                
                plt.tight_layout()
                # タイトルとプロットが重ならないように調整
                plt.subplots_adjust(top=0.9) 
                plt.savefig(f"{rank_dir_name}/{region_name}_Rank_{rank}_Bubble_No.{set_count}.png")
                plt.close(fig)
        
                # 次のセットに向けてリセット
                current_batch_indices = []
                set_count += 1

        # channel map の図示
        # Rank毎のchannelmap保存ディレクトリ準備
        rank_dir_name = f"{region_dir}/channel_map"
        Path(rank_dir_name).mkdir(parents=True, exist_ok=True)
        
        axes_num = conv_resize_no_aug_list[0].shape[0]
        
        for index in range(len(conv_resize_no_aug_list)):
            data = conv_resize_no_aug_list[index]
            
            fig, axes = plt.subplots(1, axes_num, figsize=(axes_num*3, 1*4))
            for j, data_segment in enumerate(data):
                axes[j].imshow(data_segment, vmin=0, vmax=1, cmap="jet")
                axes[j].axis("off")
                axes[j].set_title(f"{j+1}ch", fontsize=20, color='red')
        
            # fig.suptitle(f"Data No.{index}, min={data.min():.2f}, max={data.max():.2f}", fontsize=30) 
            fig.suptitle(f"Data No.{index}, min={data.min():.2f}, max={data.max():.2f}", fontsize=30, fontweight='bold')
            plt.tight_layout()
            plt.savefig(f"{rank_dir_name}/{region_name}_Rank{rank}_channel_map_No{index}.png")
            plt.close(fig)

        # ゼロで埋めたデータの削除
        deleted_indices = []
        process_end_list = []
        
        for idx, data in enumerate(conv_resize_list):
            if np.all((data == 0) | np.isnan(data)):
                deleted_indices.append(idx)
                continue
        
            process_end_list.append(data)
        
        # 削除されたインデックスを表示
        print("削除されたデータのインデックス:", deleted_indices)
        print("削除されたデータの数: ", len(deleted_indices))
        print("残ったデータ数: ",len(process_end_list))
        
        # データの保存
        np.save(f"{region_dir}/slide_bubble_data", conv_resize_list)
        np.save(f"{region_dir}/bubble_data", conv_resize_no_aug_list)