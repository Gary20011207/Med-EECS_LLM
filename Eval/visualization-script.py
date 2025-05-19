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
from typing import List, Dict, Any, Optional, Tuple
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

    Args:
        result_files: 結果檔案路徑列表

    Returns:
        Tuple[List[pd.DataFrame], List[str]]: 資料框列表和模型名稱列表
    """
    dataframes = []
    model_names = []

    for file_path in result_files:
        try:
            if not os.path.exists(file_path):
                print(f"警告：找不到檔案 {file_path}")
                continue

            df = pd.read_csv(file_path)
            dataframes.append(df)

            file_name = os.path.basename(file_path)
            match = re.search(r'([^_]+)_(with_rag|no_rag)', file_name)
            if match:
                model_name = f"{match.group(1)} ({'RAG' if match.group(2) == 'with_rag' else '無 RAG'})"
            else:
                model_name = os.path.splitext(file_name)[0]
            model_names.append(model_name)

            print(f"已載入 {file_name}，共 {len(df)} 個結果項目")

        except Exception as e:
            print(f"載入 {file_path} 時發生錯誤：{e}")

    return dataframes, model_names

def plot_metrics_overview(dataframes: List[pd.DataFrame], model_names: List[str], output_dir: str):
    """
    繪製模型指標總覽圖

    Args:
        dataframes: 結果資料框列表
        model_names: 模型名稱列表
        output_dir: 輸出目錄
    """
    metrics = ["bleu-1", "bleu-2", "bleu-3", "bleu-4",
               "rouge-1", "rouge-2", "rouge-l", "meteor"]

    fig, ax = plt.subplots(figsize=(12, 8)) # Use subplots for better control

    bar_width = 0.15
    x = np.arange(len(metrics))

    for i, (df, model_name) in enumerate(zip(dataframes, model_names)):
        avg_metrics = [df[metric].mean() for metric in metrics]
        ax.bar(x + i * bar_width, avg_metrics, bar_width,
               label=model_name, color=COLORS["category_colors"][i % len(COLORS["category_colors"])])

    ax.set_xlabel('評估指標', fontsize=12, fontweight='bold')
    ax.set_ylabel('平均分數', fontsize=12, fontweight='bold')
    ax.set_title('模型評估指標總覽', fontsize=16, fontweight='bold')
    ax.set_xticks(x + bar_width * (len(dataframes) - 1) / 2)
    ax.set_xticklabels(metrics, rotation=30, ha="right")
    ax.legend()

    plt.tight_layout()
    plt.savefig(f"{output_dir}/metrics_overview.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已儲存指標總覽圖表至 {output_dir}/metrics_overview.png")

def plot_response_time(dataframes: List[pd.DataFrame], model_names: List[str], output_dir: str):
    """
    繪製回應時間盒鬚圖 (模型在X軸，時間在Y軸)

    Args:
        dataframes: 結果資料框列表
        model_names: 模型名稱列表
        output_dir: 輸出目錄
    """
    plt.figure(figsize=(max(8, len(model_names) * 1.5), 6)) # 根據模型數量動態調整寬度

    response_times_data = []
    valid_model_names_for_plot = []

    for i, df in enumerate(dataframes):
        if "response_time" in df.columns:
            # 為每個模型的響應時間創建一個Series/list，並保留模型名稱
            times = df["response_time"].dropna()
            if not times.empty:
                # Seaborn boxplot可以直接接收一個包含多列的DataFrame，或者一個list of lists/arrays
                # 為了清晰，我們將每個模型的數據分別添加
                response_times_data.append(times.values)
                valid_model_names_for_plot.append(model_names[i])
        else:
            print(f"警告：模型 {model_names[i]} 的結果缺少 'response_time' 欄位。")

    if not response_times_data:
        print("無可用的回應時間數據進行繪圖。")
        plt.close()
        return

    # 使用 sns.boxplot，讓模型名稱在 X 軸
    # data 參數可以直接傳入 list of arrays/lists
    sns.boxplot(data=response_times_data, palette=COLORS["category_colors"][:len(valid_model_names_for_plot)])

    plt.ylabel('回應時間 (秒)', fontsize=12, fontweight='bold') # Y 軸現在是時間
    plt.xlabel('模型', fontsize=12, fontweight='bold')         # X 軸現在是模型
    plt.title('模型回應時間比較', fontsize=16, fontweight='bold')
    
    # 設定 X 軸的刻度標籤為模型名稱
    plt.xticks(ticks=np.arange(len(valid_model_names_for_plot)), # 刻度位置
               labels=valid_model_names_for_plot,             # 刻度標籤
               rotation=30,                                   # 旋轉標籤以防重疊
               ha='right')                                    # 水平對齊方式

    plt.grid(True, axis='y', linestyle='--') # 只在 Y 軸上顯示網格線可能更清晰

    plt.tight_layout() # 再次嘗試 tight_layout
    plt.savefig(f"{output_dir}/response_time_comparison.png", dpi=300, bbox_inches='tight') # 可以改個檔名以區分
    plt.close()
    print(f"已儲存回應時間比較圖表至 {output_dir}/response_time_comparison.png")

def plot_category_performance(dataframes: List[pd.DataFrame], model_names: List[str], output_dir: str):
    """
    繪製不同分類下的效能比較圖

    Args:
        dataframes: 結果資料框列表
        model_names: 模型名稱列表
        output_dir: 輸出目錄
    """
    if not all('category' in df.columns for df in dataframes):
        print("警告：部分結果檔案缺少 'category' 欄位，跳過分類效能分析")
        return

    all_categories_set = set()
    for df in dataframes:
        all_categories_set.update(df['category'].dropna().astype(str).unique())

    if not all_categories_set:
        print("警告：找不到有效的分類資料，跳過分類效能分析")
        return
        
    all_display_categories = sorted(list(all_categories_set))

    metrics = ["bleu-1", "rouge-l", "meteor"]

    for metric in metrics:
        plt.figure(figsize=(max(12, len(all_display_categories) * 0.8), 7)) # Dynamic width

        category_metric_data = {} # {model_name: [mean_scores_for_each_category]}

        for df, model_name in zip(dataframes, model_names):
            current_model_means = []
            df['category_str'] = df['category'].astype(str) # Ensure string type for comparison
            for display_cat_name in all_display_categories:
                category_df = df[df['category_str'] == display_cat_name]
                if not category_df.empty and metric in category_df.columns:
                    current_model_means.append(category_df[metric].mean())
                else:
                    current_model_means.append(0) # Or np.nan, but 0 is simpler for bar chart
            category_metric_data[model_name] = current_model_means
        
        num_models = len(dataframes)
        x = np.arange(len(all_display_categories))
        bar_width = 0.8 / num_models if num_models > 0 else 0.8

        for i, model_name in enumerate(model_names):
            means = category_metric_data.get(model_name, [0]*len(all_display_categories))
            offset = (i - num_models / 2 + 0.5) * bar_width
            plt.bar(x + offset, means, width=bar_width, label=model_name,
                    color=COLORS["category_colors"][i % len(COLORS["category_colors"])])

        plt.xlabel('分類', fontsize=12, fontweight='bold')
        plt.ylabel(f'{metric} 分數', fontsize=12, fontweight='bold')
        plt.title(f'{metric} 各分類效能比較', fontsize=16, fontweight='bold')
        plt.xticks(x, all_display_categories, rotation=45, ha='right', fontsize=10)
        plt.legend(title='模型')
        plt.grid(True, axis='y', linestyle='--')
        plt.ylim(bottom=0) # Ensure y-axis starts at 0 for scores

        plt.tight_layout()
        plt.savefig(f"{output_dir}/category_{metric}_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"已儲存分類 {metric} 效能圖表至 {output_dir}/category_{metric}_comparison.png")


def plot_response_length_analysis(dataframes: List[pd.DataFrame], model_names: List[str], output_dir: str):
    """
    分析回應長度分佈及其與指標的關係

    Args:
        dataframes: 結果資料框列表
        model_names: 模型名稱列表
        output_dir: 輸出目錄
    """
    has_length_data = any("response_length" in df.columns for df in dataframes)
    if not has_length_data:
        print("警告：無 'response_length' 欄位可供分析，跳過回應長度分析。")
        return

    plt.figure(figsize=(10, 6))
    for i, (df, model_name) in enumerate(zip(dataframes, model_names)):
        if "response_length" in df.columns:
            sns.histplot(df["response_length"].dropna(), kde=True,
                         color=COLORS["category_colors"][i % len(COLORS["category_colors"])],
                         alpha=0.6, label=model_name, stat="density", common_norm=False) # Use density for better comparison
    plt.xlabel('回應長度 (字元)', fontsize=12, fontweight='bold')
    plt.ylabel('密度', fontsize=12, fontweight='bold') # Changed from Frequency to Density
    plt.title('回應長度分佈', fontsize=16, fontweight='bold')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/response_length_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已儲存回應長度分佈圖表至 {output_dir}/response_length_distribution.png")

    metrics = ["bleu-1", "rouge-l", "meteor"]
    for i, (df, model_name) in enumerate(zip(dataframes, model_names)):
        if "response_length" not in df.columns:
            continue
        
        valid_metrics_for_model = [m for m in metrics if m in df.columns]
        if not valid_metrics_for_model:
            continue

        fig, axes = plt.subplots(1, len(valid_metrics_for_model), figsize=(5 * len(valid_metrics_for_model), 5), squeeze=False)
        axes_flat = axes.flatten()

        for j, metric in enumerate(valid_metrics_for_model):
            # Drop NaNs for correlation and plotting
            plot_data = df[["response_length", metric]].dropna()
            if plot_data.empty or len(plot_data) < 2: # Need at least 2 points for polyfit and correlation
                axes_flat[j].text(0.5, 0.5, '數據不足', horizontalalignment='center', verticalalignment='center', transform=axes_flat[j].transAxes)
                axes_flat[j].set_title(f'回應長度 vs {metric}', fontsize=12)
                continue

            axes_flat[j].scatter(plot_data["response_length"], plot_data[metric],
                                 alpha=0.5, color=COLORS["category_colors"][i % len(COLORS["category_colors"])])
            
            try:
                m_slope, b_intercept = np.polyfit(plot_data["response_length"], plot_data[metric], 1)
                axes_flat[j].plot(plot_data["response_length"], m_slope * plot_data["response_length"] + b_intercept,
                                  color='red', linestyle='--', linewidth=1)
                corr = plot_data["response_length"].corr(plot_data[metric])
                axes_flat[j].annotate(f'相關係數: {corr:.3f}',
                                      xy=(0.05, 0.95), xycoords='axes fraction',
                                      fontsize=9, ha='left', va='top',
                                      bbox=dict(boxstyle='round,pad=0.3', fc='aliceblue', alpha=0.7))
            except (np.linalg.LinAlgError, ValueError) as e: # Catch potential errors in polyfit or corr
                 axes_flat[j].annotate(f'計算錯誤: {e}',
                                      xy=(0.05, 0.95), xycoords='axes fraction',
                                      fontsize=9, ha='left', va='top', color='red')


            axes_flat[j].set_xlabel('回應長度', fontsize=10)
            axes_flat[j].set_ylabel(metric, fontsize=10)
            axes_flat[j].set_title(f'回應長度 vs {metric}', fontsize=12)

        plt.suptitle(f'{model_name} - 回應長度與指標關係', fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust for suptitle
        plt.savefig(f"{output_dir}/{model_name.replace(' ', '_').replace('(', '').replace(')', '')}_length_metrics.png",
                    dpi=300, bbox_inches='tight')
        plt.close()
        print(f"已儲存回應長度與指標關係圖表至 {output_dir}/{model_name.replace(' ', '_').replace('(', '').replace(')', '')}_length_metrics.png")


def plot_reference_analysis(dataframes: List[pd.DataFrame], model_names: List[str], output_dir: str):
    """
    分析 RAG 參考資源使用情況

    Args:
        dataframes: 結果資料框列表
        model_names: 模型名稱列表
        output_dir: 輸出目錄
    """
    if not any('used_rag_resources' in df.columns for df in dataframes): # Check if any df has the column
        print("警告：所有結果檔案均缺少 'used_rag_resources' 欄位，跳過 RAG 參考分析")
        return

    for i, (df, model_name) in enumerate(zip(dataframes, model_names)):
        if 'used_rag_resources' not in df.columns:
            print(f"警告：模型 '{model_name}' 的結果缺少 'used_rag_resources' 欄位。")
            continue
        if "with_rag" not in model_name.lower() and "rag" not in model_name.lower(): # Broader check for RAG
            continue

        all_resources = []
        for resources_str in df["used_rag_resources"].dropna():
            if isinstance(resources_str, str) and resources_str:
                cleaned_resources = [r.strip() for r in resources_str.split(',') if r.strip()]
                all_resources.extend(cleaned_resources)

        if not all_resources:
            print(f"模型 '{model_name}' 未找到有效的 RAG 參考資源數據。")
            continue
            
        resource_counts = Counter(all_resources)
        
        plt.figure(figsize=(12, max(6, len(resource_counts) * 0.3))) # Dynamic height

        top_n = 15 # Show more if available
        top_resources_dict = dict(resource_counts.most_common(top_n))
        
        if not top_resources_dict:
            print(f"模型 '{model_name}' 未找到有效的 RAG 參考資源數據可供繪圖。")
            plt.close()
            continue

        resource_names = list(top_resources_dict.keys())
        counts = list(top_resources_dict.values())

        plt.barh(resource_names, counts, # Horizontal bar chart for long names
                color=COLORS["category_colors"][i % len(COLORS["category_colors"])])

        plt.ylabel('資源檔案', fontsize=12, fontweight='bold') # Swapped x and y labels
        plt.xlabel('參考次數', fontsize=12, fontweight='bold')
        plt.title(f'{model_name} - 前 {top_n}大參考資源', fontsize=16, fontweight='bold')
        plt.gca().invert_yaxis() # Display most frequent at the top

        plt.tight_layout()
        plt.savefig(f"{output_dir}/{model_name.replace(' ', '_').replace('(', '').replace(')', '')}_reference_counts.png",
                    dpi=300, bbox_inches='tight')
        plt.close()
        print(f"已儲存參考分析圖表至 {output_dir}/{model_name.replace(' ', '_').replace('(', '').replace(')', '')}_reference_counts.png")


def plot_radar_chart(dataframes: List[pd.DataFrame], model_names: List[str], output_dir: str):
    """
    繪製雷達圖比較模型在各指標上的表現

    Args:
        dataframes: 結果資料框列表
        model_names: 模型名稱列表
        output_dir: 輸出目錄
    """
    metrics = ["bleu-1", "bleu-2", "bleu-3", "bleu-4",
               "rouge-1", "rouge-2", "rouge-l", "meteor"]
    
    # Filter out metrics not present in any dataframe
    available_metrics = []
    for metric in metrics:
        if any(metric in df.columns for df in dataframes):
            available_metrics.append(metric)
    
    if not available_metrics:
        print("警告：無可用指標繪製雷達圖。")
        return
    metrics = available_metrics


    avg_values_per_model = []
    max_val_overall = 0
    for df in dataframes:
        current_model_avg_values = []
        for metric in metrics:
            if metric in df.columns:
                mean_val = df[metric].mean()
                current_model_avg_values.append(mean_val)
                if mean_val > max_val_overall:
                    max_val_overall = mean_val
            else:
                current_model_avg_values.append(0) # Or np.nan, handle accordingly
        avg_values_per_model.append(current_model_avg_values)

    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    for i, (model_avg_scores, model_name) in enumerate(zip(avg_values_per_model, model_names)):
        plot_scores = model_avg_scores[:] + model_avg_scores[:1]  # Close the values list
        ax.plot(angles, plot_scores, 'o-', linewidth=2,
                color=COLORS["category_colors"][i % len(COLORS["category_colors"])],
                label=model_name)
        ax.fill(angles, plot_scores, alpha=0.25, # Increased alpha slightly
                color=COLORS["category_colors"][i % len(COLORS["category_colors"])])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=10) # Adjusted fontsize
    
    ax.set_yticks(np.linspace(0, max_val_overall * 1.2 if max_val_overall > 0 else 1.0, 5)) # Dynamic y-ticks
    ax.set_ylim(0, max_val_overall * 1.2 if max_val_overall > 0 else 1.0)


    plt.title('模型評估指標雷達圖', fontsize=16, fontweight='bold', y=1.1) # Adjust title position
    # Position legend outside the plot for clarity
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, ncol=min(3, len(model_names)))


    plt.tight_layout() # May need adjustment with bbox_to_anchor
    plt.savefig(f"{output_dir}/metrics_radar_chart.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已儲存指標雷達圖到 {output_dir}/metrics_radar_chart.png")


def plot_failure_analysis(dataframes: List[pd.DataFrame], model_names: List[str], output_dir: str):
    """
    分析表現最差的問題以識別模型弱點

    Args:
        dataframes: 結果資料框列表
        model_names: 模型名稱列表
        output_dir: 輸出目錄
    """
    combined_metrics = ["bleu-1", "rouge-l", "meteor"]

    for i, (df_orig, model_name) in enumerate(zip(dataframes, model_names)):
        df = df_orig.copy() # Work on a copy
        
        # Filter metrics that are actually in the dataframe
        current_metrics = [m for m in combined_metrics if m in df.columns]
        if not current_metrics:
            print(f"警告：模型 '{model_name}' 缺少用於失敗分析的關鍵指標。")
            continue
        
        df['combined_score'] = df[current_metrics].mean(axis=1)
        
        # Handle cases where all scores might be NaN after mean if all input metrics were NaN for a row
        df.dropna(subset=['combined_score'], inplace=True)
        if df.empty:
            print(f"警告：模型 '{model_name}' 計算綜合分數後無有效數據。")
            continue

        worst_questions_n = 10
        worst_questions = df.nsmallest(worst_questions_n, 'combined_score')

        if worst_questions.empty:
            print(f"模型 '{model_name}' 未找到表現差的問題 (可能所有分數都很好或數據不足)。")
            continue

        # Prepare display IDs and categories (handling potential Chinese characters)
        if 'category' in df.columns:
            worst_questions['category_display'] = worst_questions['category'].fillna('未知分類').astype(str)
        else:
            worst_questions['category_display'] = '無分類資訊'
        
        if 'qa_id' in df.columns:
             worst_questions['qa_id_display'] = worst_questions['qa_id'].fillna('未知ID').astype(str)
        else: # Use index if qa_id is missing
            worst_questions['qa_id_display'] = worst_questions.index.astype(str)

        worst_questions['display_id'] = worst_questions.apply(
            lambda x: f"{x['category_display'][:10]}_{x['qa_id_display']}", # Truncate long category names for display ID
            axis=1
        )
        
        plt.figure(figsize=(max(12, len(worst_questions) * 0.8), 7)) # Dynamic width
        
        hue_column = 'category_display' if 'category' in df.columns else None
        
        sns.barplot(x='display_id', y='combined_score', data=worst_questions,
                    palette='viridis_r', hue=hue_column, dodge=False) # Use dodge=False if not stacking per category

        plt.xlabel('問題 (分類_ID)', fontsize=12, fontweight='bold')
        plt.ylabel('綜合分數', fontsize=12, fontweight='bold')
        plt.title(f'{model_name} - 表現最差的 {len(worst_questions)} 個問題', fontsize=16, fontweight='bold')
        plt.xticks(rotation=45, ha='right', fontsize=9)
        if hue_column:
            plt.legend(title='分類', bbox_to_anchor=(1.02, 1), loc='upper left')
        plt.ylim(0, max(worst_questions['combined_score'].max() * 1.1, 0.1)) # Ensure y-axis is sensible

        plt.tight_layout()
        plt.savefig(f"{output_dir}/{model_name.replace(' ', '_').replace('(', '').replace(')', '')}_worst_questions.png",
                    dpi=300, bbox_inches='tight')
        plt.close()
        print(f"已儲存最差問題分析圖表至 {output_dir}/{model_name.replace(' ', '_').replace('(', '').replace(')', '')}_worst_questions.png")

        if 'category' in df.columns and not worst_questions['category_display'].empty:
            category_counts = worst_questions['category_display'].value_counts()
            plt.figure(figsize=(10, 8)) # Increased size for pie chart
            
            # Filter small slices into "Other" if too many categories
            threshold = 0.03 # 3%
            total_count = category_counts.sum()
            small_slices = category_counts[category_counts / total_count < threshold]
            if not small_slices.empty and len(category_counts) > 5: # Only group if there are many categories
                main_slices = category_counts[category_counts / total_count >= threshold]
                other_sum = small_slices.sum()
                category_counts_display = main_slices.append(pd.Series([other_sum], index=['其他']))
            else:
                category_counts_display = category_counts

            patches, texts, autotexts = plt.pie(category_counts_display, labels=None, autopct='%1.1f%%',
                                                startangle=90, pctdistance=0.85,
                                                colors=COLORS["category_colors"][:len(category_counts_display)])
            plt.legend(patches, category_counts_display.index, loc="center left", bbox_to_anchor=(1, 0, 0.5, 1), title="分類")
            plt.axis('equal')
            plt.title(f'{model_name} - 最差問題之分類分佈', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/{model_name.replace(' ', '_').replace('(', '').replace(')', '')}_worst_categories.png",
                        dpi=300, bbox_inches='tight')
            plt.close()
            print(f"已儲存最差問題分類分佈圖表至 {output_dir}/{model_name.replace(' ', '_').replace('(', '').replace(')', '')}_worst_categories.png")


def plot_scatter_metrics(dataframes: List[pd.DataFrame], model_names: List[str], output_dir: str):
    """
    繪製散佈圖顯示兩個指標之間的關係 (例如 BLEU vs ROUGE-L)

    Args:
        dataframes: 結果資料框列表
        model_names: 模型名稱列表
        output_dir: 輸出目錄
    """
    metric_pairs = [("bleu-1", "rouge-l"), ("bleu-4", "meteor"), ("rouge-l", "meteor")] # bleu-4 is often more informative than bleu-1

    for metric_x, metric_y in metric_pairs:
        # Check if both metrics are available in at least one dataframe
        available_for_plot = False
        for df in dataframes:
            if metric_x in df.columns and metric_y in df.columns:
                available_for_plot = True
                break
        if not available_for_plot:
            print(f"警告：指標對 ({metric_x}, {metric_y}) 在所有模型中均不完整，跳過繪圖。")
            continue

        plt.figure(figsize=(10, 8))
        for i, (df, model_name) in enumerate(zip(dataframes, model_names)):
            if metric_x in df.columns and metric_y in df.columns:
                plot_data = df[[metric_x, metric_y]].dropna()
                if not plot_data.empty:
                    plt.scatter(plot_data[metric_x], plot_data[metric_y],
                                alpha=0.5, label=model_name, # Reduced alpha
                                color=COLORS["category_colors"][i % len(COLORS["category_colors"])], s=50) # Increased size
        
        plt.xlabel(f'{metric_x} 分數', fontsize=12, fontweight='bold')
        plt.ylabel(f'{metric_y} 分數', fontsize=12, fontweight='bold')
        plt.title(f'{metric_x} 與 {metric_y} 關係', fontsize=16, fontweight='bold')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.xlim(left=0) # Scores are typically non-negative
        plt.ylim(bottom=0)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/{metric_x}_vs_{metric_y}_scatter.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"已儲存 {metric_x} 與 {metric_y} 散佈圖至 {output_dir}/{metric_x}_vs_{metric_y}_scatter.png")

def main():
    # 設定中文字型
    # 請確保系統已安裝 'Noto Sans CJK TC' 或其變體，或者 Matplotlib 可以找到的任何其他支援中文的字型
    # 'Noto Sans CJK TC' (Traditional Chinese), 'Noto Sans CJK SC' (Simplified Chinese), etc.
    plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP']
    plt.rcParams['axes.unicode_minus'] = False  # 解決負號顯示問題

    # 設定風格
    plt.style.use("seaborn-v0_8-pastel")

    """主函數"""
    parser = argparse.ArgumentParser(description="ERAS QA 模型評估視覺化腳本")
    parser.add_argument("--result_files", type=str, nargs='+', required=True,
                        help="評估結果 CSV 檔案路徑 (可指定多個檔案進行比較)")
    parser.add_argument("--output_base_dir", type=str, default="./Eval/visualizations_output", # Changed default
                        help="視覺化圖表的基礎輸出目錄")
    parser.add_argument("--use_csv_name_folder", action="store_true", # Default is False unless specified
                        help="為每個結果檔案使用 CSV 名稱建立子資料夾 (僅適用於單一檔案)")
    parser.add_argument("--no_csv_name_folder", action="store_false", dest="use_csv_name_folder",
                        help="不使用 CSV 名稱建立子資料夾 (預設行為，或用於多檔案比較)")
    parser.set_defaults(use_csv_name_folder=False)


    args = parser.parse_args()

    os.makedirs(args.output_base_dir, exist_ok=True)

    dataframes, model_names = load_results(args.result_files)

    if not dataframes:
        print("錯誤：無法載入任何結果檔案")
        sys.exit(1)

    if args.use_csv_name_folder and len(args.result_files) == 1:
        csv_filename = os.path.basename(args.result_files[0])
        csv_name = os.path.splitext(csv_filename)[0]
        output_dir = os.path.join(args.output_base_dir, csv_name)
        print(f"\n輸出目錄：{output_dir} (基於 CSV 檔案名稱)")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = f"comparison_{timestamp}"
        if len(args.result_files) == 1 and not args.use_csv_name_folder: # Single file, but not using CSV name folder
             csv_filename = os.path.basename(args.result_files[0])
             csv_name = os.path.splitext(csv_filename)[0]
             folder_name = f"{csv_name}_{timestamp}" # More descriptive for single file with timestamp
        output_dir = os.path.join(args.output_base_dir, folder_name)
        if len(args.result_files) > 1:
            print(f"\n輸出目錄：{output_dir} (多檔案比較)")
        else:
            print(f"\n輸出目錄：{output_dir}")

    os.makedirs(output_dir, exist_ok=True)

    print(f"\n開始為 {len(dataframes)} 個模型產生視覺化圖表...")

    plot_metrics_overview(dataframes, model_names, output_dir)
    plot_response_time(dataframes, model_names, output_dir)
    plot_category_performance(dataframes, model_names, output_dir)
    plot_response_length_analysis(dataframes, model_names, output_dir)
    plot_reference_analysis(dataframes, model_names, output_dir)
    plot_radar_chart(dataframes, model_names, output_dir)
    plot_failure_analysis(dataframes, model_names, output_dir)
    plot_scatter_metrics(dataframes, model_names, output_dir)

    print(f"\n視覺化完成！所有圖表已儲存至 {output_dir} 目錄")

if __name__ == "__main__":
    main()