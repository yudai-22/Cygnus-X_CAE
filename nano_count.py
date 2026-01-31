import json
import glob
import os

print("集計を開始します...")

total_lines = 0
files = glob.glob('**/*.ipynb', recursive=True)
count_files = 0

for f in files:
    # チェックポイント（自動バックアップ）は除外
    if '.ipynb_checkpoints' in f:
        continue

    try:
        with open(f, 'r', encoding='utf-8') as fp:
            data = json.load(fp)
            
        # コードセルの行数だけをカウント
        lines = sum(len(cell['source']) for cell in data.get('cells', []) if cell['cell_type'] == 'code')
        total_lines += lines
        count_files += 1
        
    except Exception as e:
        # エラーが出たファイルを表示（これの原因を知るため）
        print(f"⚠️ 読み込みスキップ: {f} (理由: {e})")

print("-" * 30)
print(f"対象ファイル数: {count_files} ファイル")
print(f"合計コード行数: {total_lines} 行")