import os
import shutil
from PIL import Image
import csv

def anonymize_data(source_root_dir='Images', output_root_dir='Images_anonymized'):
    """
    Anonymizes image data by renaming subfolders and cropping JPG images.

    Args:
        source_root_dir (str): The path to the root directory containing the original images.
        output_root_dir (str): The path to the directory where anonymized data will be saved.
    """
    # 檢查來源資料夾是否存在
    if not os.path.isdir(source_root_dir):
        print(f"錯誤：來源資料夾 '{source_root_dir}' 不存在。")
        return

    # 建立輸出資料夾 (如果不存在)
    os.makedirs(output_root_dir, exist_ok=True)

    mapping = []  # 用於儲存原始名稱和新名稱的對應關係
    subfolder_counter = 1

    print(f"開始處理資料夾：'{source_root_dir}'")
    print(f"匿名化後的資料將儲存於：'{output_root_dir}'")

    # 取得所有子資料夾名稱並排序，確保處理順序一致性
    original_subdirs = sorted([d for d in os.listdir(source_root_dir) if os.path.isdir(os.path.join(source_root_dir, d))])

    for original_subdir_name in original_subdirs:
        original_subdir_path = os.path.join(source_root_dir, original_subdir_name)

        # 1. 產生新的匿名資料夾名稱
        anonymized_subdir_name = f"sub{subfolder_counter:03d}" # 例如：sub001, sub002
        anonymized_subdir_path = os.path.join(output_root_dir, anonymized_subdir_name)
        os.makedirs(anonymized_subdir_path, exist_ok=True)

        # 記錄對應關係
        mapping.append({'original_name': original_subdir_name, 'anonymized_name': anonymized_subdir_name})
        print(f"\n正在處理資料夾 '{original_subdir_name}' -> '{anonymized_subdir_name}'")

        # 遍歷原始子資料夾中的所有檔案
        for filename in os.listdir(original_subdir_path):
            original_file_path = os.path.join(original_subdir_path, filename)
            anonymized_file_path = os.path.join(anonymized_subdir_path, filename)

            if filename.lower().endswith('.jpg'):
                try:
                    img = Image.open(original_file_path)
                    width, height = img.size
                    
                    # 2. 移除圖片上緣 1/5
                    # (left, upper, right, lower)
                    # upper 是頂部開始裁切的位置，所以是 height // 5
                    # lower 保持原來的 height
                    crop_box = (0, height // 5, width, height)
                    cropped_img = img.crop(crop_box)
                    
                    cropped_img.save(anonymized_file_path)
                    print(f"  - 已裁切並儲存圖片：'{filename}'")
                except Exception as e:
                    print(f"  - 錯誤：處理圖片 '{filename}' 失敗：{e}")
                    # 如果圖片處理失敗，可以選擇複製原圖或跳過
                    try:
                        shutil.copy2(original_file_path, anonymized_file_path)
                        print(f"  - 注意：圖片 '{filename}' 處理失敗，已複製原圖。")
                    except Exception as copy_e:
                        print(f"  - 錯誤：複製原圖 '{filename}' 也失敗：{copy_e}")

            elif filename.lower() == 'report.txt': # 或其他需要複製的檔案
                try:
                    shutil.copy2(original_file_path, anonymized_file_path)
                    print(f"  - 已複製檔案：'{filename}'")
                except Exception as e:
                    print(f"  - 錯誤：複製檔案 '{filename}' 失敗：{e}")
            else:
                # 如果有其他類型的檔案，可以選擇複製或忽略
                try:
                    shutil.copy2(original_file_path, anonymized_file_path)
                    print(f"  - 已複製其他檔案：'{filename}' (未做特殊處理)")
                except Exception as e:
                    print(f"  - 錯誤：複製其他檔案 '{filename}' 失敗：{e}")


        subfolder_counter += 1

    # 儲存對應表到 CSV 檔案
    mapping_file_path = os.path.join(output_root_dir, 'anonymization_mapping.csv')
    try:
        with open(mapping_file_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['original_name', 'anonymized_name']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(mapping)
        print(f"\n已儲存資料夾名稱對應表至：'{mapping_file_path}'")
    except IOError:
        print(f"錯誤：無法寫入對應表檔案 '{mapping_file_path}'。")

    print("\n匿名化處理完成！")

# --- 主程式執行 ---
if __name__ == "__main__":
    # 設定您的來源資料夾和目標資料夾
    # 假設您的 Images 資料夾與此 Python 腳本在同一目錄下
    source_directory = 'Images'
    output_directory = 'Images_anonymized'

    anonymize_data(source_directory, output_directory)
