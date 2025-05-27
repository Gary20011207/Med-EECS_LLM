"""
ERAS QA 模型評估視覺化腳本
用於將評估結果 CSV 檔案轉換為各種視覺化圖表
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple
import argparse
from datetime import datetime
import re
from collections import Counter

# 全局顏色配置
COLORS = {
    "primary": "#4e79a7",
    "secondary": "#f28e2b",
    "tertiary": "#59a14f",
    "quaternary": "#e15759",
    "category_colors": sns.color_palette("husl", 8)
}

def load_results(result_files: List[str]) -> Tuple[List[pd.DataFrame], List[str]]:
    """
    載入一個或多個結果檔案
    """
    dataframes = []
    model_names = []

    for file_path in result_files:
        if not os.path.exists(file_path):
            print(f"警告：找不到檔案 {file_path}")
            continue

        try:
            df = pd.read_csv(file_path)
            dataframes.append(df)

            file_name = os.path.basename(file_path)
            match = re.search(r'([^_]+)_(with_rag|no_rag)', file_name)
            if match:
                model_name = f"{match.group(1)} ({'RAG' if match.group(2) == 'with_rag' else '無 RAG'})"
            else:
                model_name = os.path.splitext(file_name)[0]
            model_names.append(model_name)

            print(f"已載入 {file_name}，共 {len(df)} 筆")

        except Exception as e:
            print(f"載入失敗 {file_path}：{e}")

    return dataframes, model_names

def plot_response_time(dataframes: List[pd.DataFrame], model_names: List[str], output_dir: str):
    """
    繪製回應時間盒鬚圖 (模型在X軸，時間在Y軸)
    """
    plt.figure(figsize=(max(8, len(model_names) * 1.5), 6))
    response_times_data = []
    valid_model_names_for_plot = []

    for i, df in enumerate(dataframes):
        if "response_time" in df.columns:
            times = df["response_time"].dropna()
            if not times.empty:
                response_times_data.append(times.values)
                valid_model_names_for_plot.append(model_names[i])

    if not response_times_data:
        print("無可用回應時間資料")
        return

    sns.boxplot(data=response_times_data, palette=COLORS["category_colors"][:len(valid_model_names_for_plot)])
    plt.ylabel("回應時間 (秒)", fontsize=12)
    plt.xlabel("模型", fontsize=12)
    plt.title("模型回應時間比較", fontsize=16)
    plt.xticks(ticks=np.arange(len(valid_model_names_for_plot)), labels=valid_model_names_for_plot, rotation=30, ha="right")
    plt.grid(True, axis="y", linestyle="--")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/response_time_comparison.png", dpi=300)
    plt.close()
    print(f"已儲存回應時間圖表至 {output_dir}/response_time_comparison.png")

def plot_category_performance(dataframes: List[pd.DataFrame], model_names: List[str], output_dir: str):
    """
    繪製不同分類下的 BERTScore 效能比較圖
    """
    if not all("category" in df.columns for df in dataframes):
        print("警告：部分結果檔案缺少 'category' 欄位，跳過分類效能分析")
        return

    all_categories = set()
    for df in dataframes:
        all_categories.update(df["category"].dropna().astype(str).unique())
    all_categories = sorted(list(all_categories))
    metric = "bertscore_f1"

    plt.figure(figsize=(max(12, len(all_categories) * 0.8), 7))
    x = np.arange(len(all_categories))
    bar_width = 0.8 / len(dataframes)

    for i, (df, model_name) in enumerate(zip(dataframes, model_names)):
        df["category_str"] = df["category"].astype(str)
        means = []
        for cat in all_categories:
            cat_df = df[df["category_str"] == cat]
            means.append(cat_df[metric].mean() if not cat_df.empty else 0)
        offset = (i - len(dataframes)/2 + 0.5) * bar_width
        plt.bar(x + offset, means, width=bar_width,
                label=model_name, color=COLORS["category_colors"][i % len(COLORS["category_colors"])])

    plt.xlabel("分類", fontsize=12)
    plt.ylabel("BERTScore-F1", fontsize=12)
    plt.title("各分類 BERTScore 效能比較", fontsize=16)
    plt.xticks(x, all_categories, rotation=45, ha="right")
    plt.legend(title="模型")
    plt.grid(True, axis="y", linestyle="--")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/category_bertscore_comparison.png", dpi=300)
    plt.close()
    print(f"已儲存分類效能圖至 {output_dir}/category_bertscore_comparison.png")

def plot_response_length_analysis(dataframes: List[pd.DataFrame], model_names: List[str], output_dir: str):
    """
    分析回應長度分佈及與 BERTScore 的關係
    """
    metric = "bertscore_f1"

    plt.figure(figsize=(10, 6))
    for i, (df, model_name) in enumerate(zip(dataframes, model_names)):
        if "response_length" in df.columns:
            sns.histplot(df["response_length"].dropna(), kde=True,
                         label=model_name,
                         color=COLORS["category_colors"][i % len(COLORS["category_colors"])],
                         stat="density", alpha=0.6)

    plt.xlabel("回應長度", fontsize=12)
    plt.ylabel("密度", fontsize=12)
    plt.title("回應長度分佈", fontsize=16)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/response_length_distribution.png", dpi=300)
    plt.close()

    for i, (df, model_name) in enumerate(zip(dataframes, model_names)):
        if "response_length" not in df.columns or metric not in df.columns:
            continue
        plot_data = df[["response_length", metric]].dropna()
        if plot_data.empty:
            continue

        plt.figure(figsize=(6, 5))
        plt.scatter(plot_data["response_length"], plot_data[metric],
                    alpha=0.6, color=COLORS["category_colors"][i % len(COLORS["category_colors"])])
        m, b = np.polyfit(plot_data["response_length"], plot_data[metric], 1)
        plt.plot(plot_data["response_length"], m * plot_data["response_length"] + b,
                 color="red", linestyle="--")

        plt.xlabel("回應長度", fontsize=12)
        plt.ylabel("BERTScore-F1", fontsize=12)
        plt.title(f"{model_name} 回應長度 vs BERTScore", fontsize=14)
        plt.grid(True)
        plt.tight_layout()
        file_name = model_name.replace(" ", "_").replace("(", "").replace(")", "")
        plt.savefig(f"{output_dir}/{file_name}_length_vs_bertscore.png", dpi=300)
        plt.close()
        print(f"已儲存 {model_name} 回應長度關係圖")

def plot_failure_analysis(dataframes: List[pd.DataFrame], model_names: List[str], output_dir: str):
    """
    分析表現最差的問題以識別模型弱點 (以 BERTScore-F1 為依據)
    """
    metric = "bertscore_f1"
    for i, (df, model_name) in enumerate(zip(dataframes, model_names)):
        if metric not in df.columns:
            continue

        worst = df[df[metric].notna()].nsmallest(10, metric)
        if worst.empty:
            continue

        worst["display_id"] = worst["qa_id"].astype(str) if "qa_id" in df.columns else worst.index.astype(str)
        if "category" in df.columns:
            worst["display_id"] = worst["category"].astype(str).str[:8] + "_" + worst["display_id"]

        plt.figure(figsize=(10, 6))
        sns.barplot(x="display_id", y=metric, data=worst,
                    palette="viridis", edgecolor="black")
        plt.xticks(rotation=45, ha="right")
        plt.xlabel("問題 (分類_ID)")
        plt.ylabel("BERTScore-F1")
        plt.title(f"{model_name} 最差問題分析")
        plt.tight_layout()
        file_name = model_name.replace(" ", "_").replace("(", "").replace(")", "")
        plt.savefig(f"{output_dir}/{file_name}_worst_questions.png", dpi=300)
        plt.close()
        print(f"已儲存 {model_name} 最差回答圖")

def main():
    # 設定中文字型（視系統環境而定）
    plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'Arial Unicode MS', 'Microsoft JhengHei']
    plt.rcParams['axes.unicode_minus'] = False  # 正確顯示負號

    # 設定風格
    plt.style.use("seaborn-v0_8-pastel")

    parser = argparse.ArgumentParser(description="ERAS QA 模型評估視覺化腳本")
    parser.add_argument("--result_files", type=str, nargs='+', required=True,
                        help="評估結果 CSV 檔案路徑 (可指定多個檔案進行比較)")
    parser.add_argument("--output_base_dir", type=str, default="./Eval/visualizations_output",
                        help="圖表輸出根目錄")
    parser.add_argument("--use_csv_name_folder", action="store_true",
                        help="單一檔案時使用CSV檔名建立子資料夾")
    parser.add_argument("--no_csv_name_folder", action="store_false", dest="use_csv_name_folder",
                        help="不使用CSV名稱資料夾（預設）")
    parser.set_defaults(use_csv_name_folder=False)

    args = parser.parse_args()

    os.makedirs(args.output_base_dir, exist_ok=True)

    dataframes, model_names = load_results(args.result_files)
    if not dataframes:
        print("錯誤：未成功載入任何結果檔案")
        sys.exit(1)

    # 動態決定輸出路徑
    if args.use_csv_name_folder and len(args.result_files) == 1:
        csv_name = os.path.splitext(os.path.basename(args.result_files[0]))[0]
        output_dir = os.path.join(args.output_base_dir, csv_name)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = os.path.splitext(os.path.basename(args.result_files[0]))[0]
        folder_name = base_name if len(args.result_files) == 1 else f"comparison_{timestamp}"
        output_dir = os.path.join(args.output_base_dir, folder_name)

    os.makedirs(output_dir, exist_ok=True)
    print(f"\n輸出圖表將儲存至：{output_dir}")

    print(f"處理模型數量：{len(dataframes)}")

    # 執行繪圖函式
    plot_response_time(dataframes, model_names, output_dir)
    plot_category_performance(dataframes, model_names, output_dir)
    plot_response_length_analysis(dataframes, model_names, output_dir)
    plot_failure_analysis(dataframes, model_names, output_dir)

    print("\n✅ 圖表產生完畢！")

if __name__ == "__main__":
    main()